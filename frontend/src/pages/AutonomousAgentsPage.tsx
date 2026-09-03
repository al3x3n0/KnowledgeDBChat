/**
 * Autonomous Agents Page
 *
 * Manage and monitor autonomous agent jobs that run independently
 * to accomplish goals like research, monitoring, and analysis.
 */

import React, { useState, useEffect, useLayoutEffect, useCallback, useMemo, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Bot,
  Play,
  XCircle,
  RotateCcw,
  Plus,
  Eye,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Search,
  Activity,
  BarChart3,
  FileText,
  RefreshCw,
  Zap,
  Settings,
  Layers,
  Link2,
  GitBranch,
  Download,
  FileDown,
  Brain,
  Sparkles,
  Inbox,
  Bug,
  ThumbsUp,
  ThumbsDown,
  Map as MapIcon,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import type {
  User,
  AgentCheckpointQueueItem,
  AgentCheckpointQueueAction,
  AgentDecisionTraceEvent,
  AgentDecisionTraceAnalyticsResponse,
  AgentDecisionTraceView,
  AgentJob,
  AgentJobCreate,
  AgentJobFromTemplate,
  AgentJobProgressUpdate,
  AgentJobQuickStartBugTriageSwarmRequest,
  AgentJobQuickStartBuildBreakSwarmRequest,
  AgentJobQuickStartClaudeBackendRequest,
  AgentJobQuickStartDomainResearchRequest,
  AgentJobPromoteDomainResearchRequest,
  AgentJobPromoteDomainResearchResponse,
  AgentJobQuickStartFrontendRegressionSwarmRequest,
  AgentJobQuickStartRepoBugTriageRequest,
  AgentJobQuickStartRoleWorkflowRequest,
  AgentJobSwarmOutcomeCase,
  CollaborationSummary,
  CodingBacklogItem,
  CodingBacklogItemCreate,
  CodingBacklogDecomposition,
  CodingBacklogLatestSummary,
  CodingBacklogPolicy,
  CodingSwarmProfile,
  CodingSwarmProfileCreate,
  CodingSwarmProfileUpdate,
  CodingBacklogSlice,
  CodingBacklogTimelineEntry,
  DomainResearchProfile,
  DomainResearchProfileCreate,
  DomainResearchProfileUpdate,
  ResearchOpportunity,
  ResearchPortfolio,
  ResearchPortfolioCreate,
  ResearchPortfolioUpdate,
  ScientificSandboxProfile,
  ScientificSandboxProfileCreate,
  ScientificSandboxProfileUpdate,
  AgentJobTemplate,
  AgentJobStatus,
  AgentJobType,
  AgentJobChainDefinition,
  AgentJobChainStatus,
  AgentJobFromChainCreate,
  ResearchInboxItem,
  ResearchInboxItemStatus,
  ResearchMonitorAnalyticsResponse,
  ResearchMonitorCustomerRebalanceEvaluationDetail,
  ResearchMonitorCustomerPortfolio,
  ResearchMonitorCustomerRebalancePreview,
  ResearchMonitorHealthSummary,
  ResearchMonitorPolicyHistoryEntry,
  ResearchMonitorPolicyEvaluationDetail,
  ResearchMonitorPolicySimulationResponse,
  ScientificValidationRunSummary,
} from '../types';
import {
} from '../utils/agentMemoryExtraction';
import { mergeProgressUpdateIntoJob, TERMINAL_JOB_STATUSES } from '../utils/agentJobProgress';
import {
  getExperimentRecoveryPriority as getExperimentRecoveryPriorityForRun,
  summarizeExperimentRun,
} from '../utils/experimentRunSummary';
import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';
import CreateFromTemplateModal from '../components/agent/CreateFromTemplateModal';
import CustomerResearchModal from '../components/agent/CustomerResearchModal';
import InboxMonitorModal from '../components/agent/InboxMonitorModal';
import MonitorProfilesModal from '../components/agent/MonitorProfilesModal';
import QuickStartClaudeBackendModal from '../components/agent/QuickStartClaudeBackendModal';
import QuickStartCodingSwarmModal from '../components/agent/QuickStartCodingSwarmModal';
import QuickStartRepoBugTriageModal from '../components/agent/QuickStartRepoBugTriageModal';
import QuickStartRoleWorkflowModal from '../components/agent/QuickStartRoleWorkflowModal';
import QuickStartDomainResearchModal from '../components/agent/QuickStartDomainResearchModal';
import CreateJobModal from '../components/agent/CreateJobModal';
import StartChainModal from '../components/agent/StartChainModal';
import TemplateCard from '../components/agent/TemplateCard';
import { JOB_TYPE_CONFIG, STATUS_CONFIG } from '../components/agent/jobConfig';
import { getLatestExperimentRun } from '../components/agent/jobFields';
import JobCard from '../components/agent/JobCard';
import JobDetailPanel from '../components/agent/JobDetailPanel';
import {
  humanizeSwarmOutcome,
  summarizeSchedulerState,
  swarmOutcomeBadgeClass,
} from '../utils/agentJobDetail';
import { swarmQuickStartPreset } from '../components/agent/swarmQuickStarts';
import { copyText } from '../utils/clipboard';
import {
  buildBugTriageSwarmQuickStartPayload,
  buildBuildBreakSwarmQuickStartPayload,
  buildFrontendRegressionSwarmQuickStartPayload,
  DEFAULT_VALIDATION_POLICY,
  DOMAIN_SOURCE_SCOPE_OPTIONS,
  DOMAIN_TRACK_OPTIONS,
  parseQuickStartCommands,
  parseSafeRelativeFilePaths,
  splitUniqueLines,
} from './autonomousAgentQuickStarts';

// Job type icons and labels

const AUTONOMOUS_SYSTEM_MAP = String.raw`KnowledgeDBChat
├─ Frontend UI: AutonomousAgentsPage
│  ├─ Queue
│  ├─ Health
│  ├─ Jobs
│  ├─ Domain Profiles
│  ├─ Research Fleet
│  └─ Inbox / Swarm / Backlog
├─ Backend APIs
│  ├─ agent_jobs
│  ├─ domain_research_profiles
│  ├─ research_portfolios
│  ├─ research_monitor_profiles
│  └─ experiments
├─ Core services
│  ├─ autonomy_service
│  ├─ autonomous_agent_executor
│  ├─ research_monitor_profile_service
│  ├─ research_opportunity_service
│  └─ scientific_validation_service
└─ Persistence
   ├─ AgentJob
   ├─ DomainResearchProfile
   ├─ ResearchPortfolio
   ├─ ResearchMonitorProfile
   └─ ExperimentPlan / ExperimentRun`;




const summarizeTraceAnalyticsBuckets = (items: Array<{ value: string; count: number }> | undefined | null): string[] => {
  if (!items?.length) return [];
  return items
    .filter((item) => String(item.value || '').trim())
    .slice(0, 3)
    .map((item) => `${humanizeDecisionTraceValue(item.value)} (${Number(item.count || 0)})`);
};

const formatTraceAnalyticsDay = (value: string): string => {
  const parsed = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
};

const TRACE_FILTER_QUERY_KEYS = [
  'trace_source_kind',
  'trace_decision_type',
  'trace_customer',
  'trace_status',
  'trace_severity',
  'trace_actor_mode',
  'trace_triage_status',
  'trace_assigned_to_user_id',
  'trace_unassigned_only',
  'trace_escalation_state',
  'trace_pinned',
  'trace_actionable_only',
  'trace_date_range',
] as const;

const normalizeTraceViewFilters = (filters?: Record<string, any> | null) => ({
  source_kind: String(filters?.source_kind || '').trim(),
  decision_type: String(filters?.decision_type || '').trim(),
  customer: String(filters?.customer || '').trim(),
  status: String(filters?.status || '').trim(),
  severity: String(filters?.severity || '').trim(),
  actor_mode: String(filters?.actor_mode || '').trim(),
  triage_status: String(filters?.triage_status || '').trim(),
  assigned_to_user_id: String(filters?.assigned_to_user_id || '').trim(),
  unassigned_only: Boolean(filters?.unassigned_only),
  escalation_state: String(filters?.escalation_state || '').trim(),
  pinned: Boolean(filters?.pinned),
  actionable_only: Boolean(filters?.actionable_only),
  date_range: String(filters?.date_range || '7d').trim() || '7d',
});

const traceViewFiltersMatch = (left?: Record<string, any> | null, right?: Record<string, any> | null) =>
  JSON.stringify(normalizeTraceViewFilters(left)) === JSON.stringify(normalizeTraceViewFilters(right));

const canonicalizeSearchParams = (search: string) =>
  Array.from(new URLSearchParams(search).entries())
    .map(([key, value]) => `${key}=${value}`)
    .sort()
    .join('&');


type ResearchPortfolioPolicyDraft = {
  automation_profile: 'balanced' | 'max_autonomy';
  follow_up_review_mode: 'auto_launch_safe' | 'queue_for_approval' | 'manual_only';
  confidence_threshold: string;
  experiment_readiness_threshold: string;
  max_auto_follow_up_launches: string;
  max_concurrent_validation_runs: string;
  max_validation_runtime_minutes: string;
  max_validation_budget_per_run: string;
  duplicate_window_items: string;
  auto_create_experiment_plans: boolean;
  auto_launch_follow_up: boolean;
  auto_launch_experiment_runs: boolean;
};

type DomainResearchProfilePolicyDraft = ResearchPortfolioPolicyDraft;

const buildResearchPortfolioPolicyDraft = (portfolio?: Partial<ResearchPortfolio> | null): ResearchPortfolioPolicyDraft => {
  const policy = ((portfolio?.effective_policy || portfolio?.automation_policy || {}) as Record<string, any>) || {};
  const automationProfile = String(portfolio?.automation_profile || 'balanced').trim().toLowerCase() === 'max_autonomy'
    ? 'max_autonomy'
    : 'balanced';
  return {
    automation_profile: automationProfile,
    follow_up_review_mode: (['auto_launch_safe', 'queue_for_approval', 'manual_only'].includes(String(policy.follow_up_review_mode || '').trim())
      ? String(policy.follow_up_review_mode).trim()
      : 'auto_launch_safe') as 'auto_launch_safe' | 'queue_for_approval' | 'manual_only',
    confidence_threshold: String(policy.confidence_threshold ?? (automationProfile === 'max_autonomy' ? 0.68 : 0.72)),
    experiment_readiness_threshold: String(policy.experiment_readiness_threshold ?? (automationProfile === 'max_autonomy' ? 0.72 : 0.8)),
    max_auto_follow_up_launches: String(policy.max_auto_follow_up_launches ?? (automationProfile === 'max_autonomy' ? 4 : 2)),
    max_concurrent_validation_runs: String(policy.max_concurrent_validation_runs ?? (automationProfile === 'max_autonomy' ? 2 : 1)),
    max_validation_runtime_minutes: String(policy.max_validation_runtime_minutes ?? (automationProfile === 'max_autonomy' ? 30 : 20)),
    max_validation_budget_per_run: String(policy.max_validation_budget_per_run ?? (automationProfile === 'max_autonomy' ? 50 : 25)),
    duplicate_window_items: String(policy.duplicate_window_items ?? (automationProfile === 'max_autonomy' ? 120 : 60)),
    auto_create_experiment_plans: Boolean(policy.auto_create_experiment_plans ?? true),
    auto_launch_follow_up: Boolean(policy.auto_launch_follow_up ?? true),
    auto_launch_experiment_runs: Boolean(policy.auto_launch_experiment_runs ?? (automationProfile === 'max_autonomy')),
  };
};

const buildResearchPortfolioUpdatePayload = (draft: ResearchPortfolioPolicyDraft): ResearchPortfolioUpdate => ({
  automation_profile: draft.automation_profile,
  automation_policy: {
    follow_up_review_mode: draft.follow_up_review_mode,
    confidence_threshold: Number(draft.confidence_threshold || 0),
    experiment_readiness_threshold: Number(draft.experiment_readiness_threshold || 0),
    max_auto_follow_up_launches: Number(draft.max_auto_follow_up_launches || 0),
    max_concurrent_validation_runs: Number(draft.max_concurrent_validation_runs || 0),
    max_validation_runtime_minutes: Number(draft.max_validation_runtime_minutes || 0),
    max_validation_budget_per_run: Number(draft.max_validation_budget_per_run || 0),
    duplicate_window_items: Number(draft.duplicate_window_items || 0),
    auto_create_experiment_plans: draft.auto_create_experiment_plans,
    auto_launch_follow_up: draft.auto_launch_follow_up,
    auto_launch_experiment_runs: draft.auto_launch_experiment_runs,
    auto_execute_validation_runs: draft.auto_launch_experiment_runs,
  },
});

const buildDomainResearchProfilePolicyDraft = (profile?: Partial<DomainResearchProfile> | null): DomainResearchProfilePolicyDraft => {
  const policy = ((profile?.effective_policy || profile?.automation_policy || {}) as Record<string, any>) || {};
  const automationProfile = String(profile?.automation_profile || 'balanced').trim().toLowerCase() === 'max_autonomy'
    ? 'max_autonomy'
    : 'balanced';
  return {
    automation_profile: automationProfile,
    follow_up_review_mode: (['auto_launch_safe', 'queue_for_approval', 'manual_only'].includes(String(policy.follow_up_review_mode || '').trim())
      ? String(policy.follow_up_review_mode).trim()
      : 'auto_launch_safe') as 'auto_launch_safe' | 'queue_for_approval' | 'manual_only',
    confidence_threshold: String(policy.confidence_threshold ?? profile?.confidence_threshold ?? (automationProfile === 'max_autonomy' ? 0.68 : 0.72)),
    experiment_readiness_threshold: String(policy.experiment_readiness_threshold ?? (automationProfile === 'max_autonomy' ? 0.72 : 0.8)),
    max_auto_follow_up_launches: String(policy.max_auto_follow_up_launches ?? (automationProfile === 'max_autonomy' ? 4 : 2)),
    max_concurrent_validation_runs: String(policy.max_concurrent_validation_runs ?? (automationProfile === 'max_autonomy' ? 2 : 1)),
    max_validation_runtime_minutes: String(policy.max_validation_runtime_minutes ?? (automationProfile === 'max_autonomy' ? 30 : 20)),
    max_validation_budget_per_run: String(policy.max_validation_budget_per_run ?? (automationProfile === 'max_autonomy' ? 50 : 25)),
    duplicate_window_items: String(policy.duplicate_window_items ?? (automationProfile === 'max_autonomy' ? 120 : 60)),
    auto_create_experiment_plans: Boolean(policy.auto_create_experiment_plans ?? profile?.auto_create_experiment_plans ?? true),
    auto_launch_follow_up: Boolean(policy.auto_launch_follow_up ?? profile?.auto_launch_follow_up ?? true),
    auto_launch_experiment_runs: Boolean(policy.auto_launch_experiment_runs ?? policy.auto_execute_validation_runs ?? (automationProfile === 'max_autonomy')),
  };
};

const buildDomainResearchProfileUpdatePayload = (draft: DomainResearchProfilePolicyDraft): DomainResearchProfileUpdate => ({
  automation_profile: draft.automation_profile,
  automation_policy: {
    follow_up_review_mode: draft.follow_up_review_mode,
    confidence_threshold: Number(draft.confidence_threshold || 0),
    experiment_readiness_threshold: Number(draft.experiment_readiness_threshold || 0),
    max_auto_follow_up_launches: Number(draft.max_auto_follow_up_launches || 0),
    max_concurrent_validation_runs: Number(draft.max_concurrent_validation_runs || 0),
    max_validation_runtime_minutes: Number(draft.max_validation_runtime_minutes || 0),
    max_validation_budget_per_run: Number(draft.max_validation_budget_per_run || 0),
    duplicate_window_items: Number(draft.duplicate_window_items || 0),
    auto_create_experiment_plans: draft.auto_create_experiment_plans,
    auto_launch_follow_up: draft.auto_launch_follow_up,
    auto_launch_experiment_runs: draft.auto_launch_experiment_runs,
    auto_execute_validation_runs: draft.auto_launch_experiment_runs,
  },
});


const formatAutonomyLabel = (value?: string | null) => String(value || 'balanced').replace(/_/g, ' ');
const formatReviewModeLabel = (value?: string | null) => String(value || 'auto_launch_safe').replace(/_/g, ' ');
const canonicalReviewModeFromMonitor = (monitor?: Partial<ResearchMonitorHealthSummary> | null) =>
  String((monitor?.effective_policy || {})?.follow_up_review_mode || (monitor?.automation_policy || {})?.follow_up_review_mode || monitor?.current_policy_mode || 'manual_only');
const canonicalAllowedRecommendationsFromMonitor = (monitor?: Partial<ResearchMonitorHealthSummary> | null) => {
  const effective = (monitor?.effective_policy || {}) as Record<string, any>;
  const automation = (monitor?.automation_policy || {}) as Record<string, any>;
  if (Array.isArray(effective.allowed_recommendations)) {
    return effective.allowed_recommendations as string[];
  }
  if (Array.isArray(automation.allowed_recommendations)) {
    return automation.allowed_recommendations as string[];
  }
  if (Array.isArray(monitor?.current_allowed_recommendations)) {
    return monitor.current_allowed_recommendations as string[];
  }
  return ['deep_dive_chain', 'single_research_job'];
};
const canonicalReviewModeFromMonitorPolicyHistoryEntry = (
  entry?: Partial<ResearchMonitorPolicyHistoryEntry> | null,
  phase: 'previous' | 'next' = 'next'
) =>
  String(
    (((phase === 'next' ? entry?.next_effective_policy : entry?.previous_effective_policy) || {}) as Record<string, any>).follow_up_review_mode
    || (((phase === 'next' ? entry?.next_automation_policy : entry?.previous_automation_policy) || {}) as Record<string, any>).follow_up_review_mode
    || (phase === 'next' ? entry?.next_follow_up_autonomy?.mode : entry?.previous_follow_up_autonomy?.mode)
    || 'manual_only'
  );
const canonicalAllowedRecommendationsFromMonitorPolicyHistoryEntry = (
  entry?: Partial<ResearchMonitorPolicyHistoryEntry> | null,
  phase: 'previous' | 'next' = 'next'
) => {
  const effective = (((phase === 'next' ? entry?.next_effective_policy : entry?.previous_effective_policy) || {}) as Record<string, any>) || {};
  const automation = (((phase === 'next' ? entry?.next_automation_policy : entry?.previous_automation_policy) || {}) as Record<string, any>) || {};
  if (Array.isArray(effective.allowed_recommendations)) {
    return effective.allowed_recommendations as string[];
  }
  if (Array.isArray(automation.allowed_recommendations)) {
    return automation.allowed_recommendations as string[];
  }
  return ((phase === 'next' ? entry?.next_follow_up_autonomy?.allowed_recommendations : entry?.previous_follow_up_autonomy?.allowed_recommendations) || []) as string[];
};

const AutonomyStatCard: React.FC<{
  label: string;
  value: React.ReactNode;
  detail?: React.ReactNode;
}> = ({ label, value, detail }) => (
  <div className="bg-white border border-gray-200 rounded p-2">
    <div className="text-gray-500">{label}</div>
    <div className="mt-1 font-medium text-gray-900">{value}</div>
    {detail ? <div className="text-gray-500">{detail}</div> : null}
  </div>
);

const SharedAutonomyMetricGrid: React.FC<{
  columns?: string;
  items: Array<{ label: string; value: React.ReactNode; detail?: React.ReactNode }>;
}> = ({ columns = 'grid-cols-4', items }) => (
  <div className={`grid ${columns} gap-2`}>
    {items.map((item) => (
      <AutonomyStatCard key={item.label} label={item.label} value={item.value} detail={item.detail} />
    ))}
  </div>
);

const SharedPortfolioLikeAutonomyControls: React.FC<{
  draft: ResearchPortfolioPolicyDraft;
  applyLabel: string;
  disabled?: boolean;
  onApply: () => void;
  onFieldChange: (field: keyof ResearchPortfolioPolicyDraft, value: any) => void;
}> = ({ draft, applyLabel, disabled, onApply, onFieldChange }) => (
  <div className="bg-white border border-gray-200 rounded p-2">
    <div className="flex items-center justify-between gap-2">
      <div className="font-medium text-gray-800">Autonomy controls</div>
      <Button size="sm" variant="secondary" onClick={onApply} disabled={disabled}>
        {applyLabel}
      </Button>
    </div>
    <div className="mt-2 grid grid-cols-2 gap-2">
      <label className="text-gray-600">
        Autonomy profile
        <select
          className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs"
          value={draft.automation_profile}
          onChange={(e) => onFieldChange('automation_profile', e.target.value as 'balanced' | 'max_autonomy')}
        >
          <option value="balanced">balanced</option>
          <option value="max_autonomy">max autonomy</option>
        </select>
      </label>
      <label className="text-gray-600">
        Review mode
        <select
          className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs"
          value={draft.follow_up_review_mode}
          onChange={(e) => onFieldChange('follow_up_review_mode', e.target.value as 'auto_launch_safe' | 'queue_for_approval' | 'manual_only')}
        >
          <option value="auto_launch_safe">auto launch safe</option>
          <option value="queue_for_approval">queue for approval</option>
          <option value="manual_only">manual only</option>
        </select>
      </label>
      <label className="text-gray-600">
        Confidence threshold
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.confidence_threshold} onChange={(e) => onFieldChange('confidence_threshold', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Readiness threshold
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.experiment_readiness_threshold} onChange={(e) => onFieldChange('experiment_readiness_threshold', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Follow-up cap
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.max_auto_follow_up_launches} onChange={(e) => onFieldChange('max_auto_follow_up_launches', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Validation concurrency
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.max_concurrent_validation_runs} onChange={(e) => onFieldChange('max_concurrent_validation_runs', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Duplicate window
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.duplicate_window_items} onChange={(e) => onFieldChange('duplicate_window_items', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Runtime minutes
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.max_validation_runtime_minutes} onChange={(e) => onFieldChange('max_validation_runtime_minutes', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Budget per run
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.max_validation_budget_per_run} onChange={(e) => onFieldChange('max_validation_budget_per_run', e.target.value)} />
      </label>
    </div>
    <div className="mt-2 flex flex-wrap gap-3 text-gray-600">
      <label className="inline-flex items-center gap-2">
        <input type="checkbox" checked={draft.auto_create_experiment_plans} onChange={(e) => onFieldChange('auto_create_experiment_plans', e.target.checked)} />
        Auto-create plans
      </label>
      <label className="inline-flex items-center gap-2">
        <input type="checkbox" checked={draft.auto_launch_follow_up} onChange={(e) => onFieldChange('auto_launch_follow_up', e.target.checked)} />
        Auto-launch follow-up
      </label>
      <label className="inline-flex items-center gap-2">
        <input type="checkbox" checked={draft.auto_launch_experiment_runs} onChange={(e) => onFieldChange('auto_launch_experiment_runs', e.target.checked)} />
        Auto-launch validation
      </label>
    </div>
  </div>
);

const CollaborationSummaryPanel: React.FC<{
  summary?: CollaborationSummary | null;
  fallbackOwnerId?: string | null;
  fallbackVisibility?: string | null;
  fallbackSharedWithUserIds?: string[];
  userLabelById: (userId: string) => string;
  assigneeUsers?: User[];
  showAssigneeSelect?: boolean;
  assigneeValue?: string;
  onAssigneeChange?: (value: string) => void;
  onClearAssignee?: () => void;
  noteValue?: string;
  onNoteChange?: (value: string) => void;
  onNoteSave?: () => void;
  noteSaveLabel?: string;
  notePlaceholder?: string;
}> = ({
  summary,
  fallbackOwnerId,
  fallbackVisibility,
  fallbackSharedWithUserIds = [],
  userLabelById,
  assigneeUsers = [],
  showAssigneeSelect = false,
  assigneeValue,
  onAssigneeChange,
  onClearAssignee,
  noteValue,
  onNoteChange,
  onNoteSave,
  noteSaveLabel = 'Save note',
  notePlaceholder = 'Add a note',
}) => {
  const ownerId = String(summary?.owner_user_id || fallbackOwnerId || '').trim();
  const assigneeId = String(summary?.assigned_user_id || assigneeValue || '').trim();
  const assignedById = String(summary?.assigned_by_user_id || '').trim();
  const sharedWithUserIds = Array.isArray(summary?.shared_with_user_ids)
    ? summary.shared_with_user_ids.map((value) => String(value || '').trim()).filter(Boolean)
    : fallbackSharedWithUserIds.map((value) => String(value || '').trim()).filter(Boolean);
  const visibilityScope = String(summary?.visibility_scope || fallbackVisibility || (sharedWithUserIds.length > 0 ? 'shared' : 'private')).trim() || 'private';
  const ownerLabel = String(summary?.owner_label || (ownerId ? userLabelById(ownerId) : '') || ownerId || 'n/a').trim();
  const assigneeLabel = String(summary?.assignee_label || (assigneeId ? userLabelById(assigneeId) : '') || assigneeId || '').trim();
  const assignedByLabel = String(assignedById ? userLabelById(assignedById) : '').trim();
  const noteText = String(noteValue ?? summary?.note ?? '').trim();
  const assigneeList = assigneeUsers.length > 0 ? assigneeUsers : [];

  return (
    <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="flex flex-wrap gap-2 text-xs text-slate-700">
        <span>Owner {ownerLabel}</span>
        {assigneeLabel ? <span>Assignee {assigneeLabel}</span> : null}
        {assignedByLabel ? <span>Assigned by {assignedByLabel}</span> : null}
        <span>Visibility {humanizeDecisionTraceValue(visibilityScope)}</span>
        {sharedWithUserIds.length > 0 ? <span>Shared with {sharedWithUserIds.length}</span> : null}
      </div>
      {showAssigneeSelect && onAssigneeChange ? (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <select
            className="border border-gray-300 rounded-lg px-2 py-1 text-xs"
            value={assigneeValue ?? assigneeId}
            onChange={(e) => onAssigneeChange(String(e.target.value || '').trim())}
          >
            <option value="">Unassigned</option>
            {assigneeList.map((candidate) => (
              <option key={String(candidate.id)} value={String(candidate.id)}>
                {userLabelById(String(candidate.id))}
              </option>
            ))}
          </select>
          {onClearAssignee ? (
            <Button size="sm" variant="ghost" onClick={onClearAssignee} disabled={!assigneeId}>
              Clear assignment
            </Button>
          ) : null}
        </div>
      ) : null}
      {onNoteChange ? (
        <div className="mt-2 space-y-2">
          <textarea
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
            rows={2}
            placeholder={notePlaceholder}
            value={noteText}
            onChange={(e) => onNoteChange(e.target.value)}
          />
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs text-slate-600">Note {noteText ? 'saved locally until you click save' : 'optional'}</div>
            {onNoteSave ? (
              <Button size="sm" variant="ghost" onClick={onNoteSave}>
                {noteSaveLabel}
              </Button>
            ) : null}
          </div>
        </div>
      ) : noteText ? (
        <div className="mt-2 text-xs text-slate-600">Note: {noteText}</div>
      ) : null}
    </div>
  );
};

const SharedAutonomyReviewLists: React.FC<{
  sections: Array<{
    title: string;
    rows?: Array<Record<string, any>> | null;
    formatter?: (row: Record<string, any>) => React.ReactNode;
    renderRow?: (row: Record<string, any>, idx: number) => React.ReactNode;
    limit?: number;
  }>;
}> = ({ sections }) => (
  <>
    {sections
      .filter((section) => Array.isArray(section.rows) && section.rows.length > 0)
      .map((section) => (
        <div key={section.title} className="bg-white border border-gray-200 rounded p-2">
          <div className="font-medium text-gray-800">{section.title}</div>
          <div className="mt-1 space-y-1">
            {(section.rows || []).slice(0, section.limit || 4).map((row, idx) => {
              const key = `${String(row.opportunity_id || row.canonical_key || idx)}`;
              if (section.renderRow) {
                return (
                  <React.Fragment key={key}>
                    {section.renderRow(row, idx)}
                  </React.Fragment>
                );
              }
              return (
                <div key={key} className="text-gray-600">
                  {section.formatter
                    ? section.formatter(row)
                    : `${String(row.title || row.canonical_key || 'Opportunity')} · ${String(row.reason_code || row.review_type || 'review').replaceAll('_', ' ')}`}
                </div>
              );
            })}
          </div>
        </div>
      ))}
  </>
);


const scientificResearchPackBlueprint = (repoSourceIds: string[]) => {
  const sourceScope = repoSourceIds.length > 0 ? 'kb_plus_arxiv_plus_repo' : 'kb_plus_arxiv';
  return {
    sourceScope,
    compiler: {
      title: 'Compiler Research Pack',
      domain: 'Compiler optimization and code generation',
      objective: 'Identify evidence-backed compiler opportunities, regressions, and validation experiments for the customer codebase.',
      track_type: 'compiler' as const,
      monitor_queries: [
        'llvm optimization pass regression',
        'mlir codegen scheduling',
        'auto-vectorization blocker benchmark',
      ],
      benchmark_queries: [
        'compile time regression',
        'vectorization benchmark',
        'codegen hotspot',
      ],
    },
    microarchitecture: {
      title: 'Microarchitecture Research Pack',
      domain: 'CPU microarchitecture performance and bottlenecks',
      objective: 'Surface testable microarchitecture opportunities tied to cache behavior, branch behavior, SIMD usage, and benchmark regressions.',
      track_type: 'microarchitecture' as const,
      monitor_queries: [
        'cache miss bottleneck benchmark',
        'branch predictor workload analysis',
        'simd throughput regression',
      ],
      benchmark_queries: [
        'ipc stall benchmark',
        'branch miss benchmark',
        'memory bandwidth benchmark',
      ],
    },
    portfolio: {
      title: 'Scientific Research Fleet',
      objective: 'Continuously rank novel and testable compiler and microarchitecture ideas, auto-create validation plans, and auto-launch bounded deep dives within budget.',
    },
  };
};

const humanizeScientificValidationReason = (value?: string | null) =>
  String(value || '').trim().replaceAll('_', ' ') || 'unknown reason';

const humanizeDecisionTraceValue = (value?: string | null) =>
  String(value || '').trim().replaceAll('_', ' ') || 'unknown';

const decisionTraceSeverityClasses = (value?: string | null) => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'high') return 'bg-rose-100 text-rose-700';
  if (normalized === 'medium') return 'bg-amber-100 text-amber-800';
  return 'bg-slate-100 text-slate-700';
};

const decisionTraceTriageClasses = (value?: string | null) => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'resolved') return 'bg-emerald-100 text-emerald-700';
  if (normalized === 'investigating') return 'bg-blue-100 text-blue-700';
  if (normalized === 'acknowledged') return 'bg-amber-100 text-amber-800';
  return 'bg-rose-100 text-rose-700';
};

const AUTONOMY_FOCUS_ROW_CLASS = 'border-cyan-300 bg-cyan-50 ring-2 ring-cyan-200';
const AUTONOMY_FOCUS_CARD_CLASS = 'border-cyan-300 ring-2 ring-cyan-200';
const HEALTH_FOCUS_CARD_CLASS = 'border-cyan-300 bg-cyan-50 ring-2 ring-cyan-200';

const toDecisionTraceDueInputValue = (value?: string | null) => {
  if (!value) return '';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return '';
  const year = parsed.getFullYear();
  const month = String(parsed.getMonth() + 1).padStart(2, '0');
  const day = String(parsed.getDate()).padStart(2, '0');
  const hours = String(parsed.getHours()).padStart(2, '0');
  const minutes = String(parsed.getMinutes()).padStart(2, '0');
  return `${year}-${month}-${day}T${hours}:${minutes}`;
};

const canActOnTraceFollowUpEvent = (event: AgentDecisionTraceEvent) => {
  if (event.is_derived) return false;
  const eventKind = String(event.event_type || event.decision_type || '').trim().toLowerCase();
  if (!['follow_up_queued', 'follow_up_queued_for_approval'].includes(eventKind)) return false;
  const sourceKind = String(event.source_kind || '').trim().toLowerCase();
  if (!['domain_profile', 'portfolio'].includes(sourceKind)) return false;
  const metadata = (event.metadata && typeof event.metadata === 'object') ? (event.metadata as Record<string, any>) : {};
  const deepLinkParams = (event.deep_link?.params && typeof event.deep_link.params === 'object')
    ? (event.deep_link.params as Record<string, any>)
    : {};
  const opportunityId = String(
    metadata.profile_opportunity_id
    || metadata.portfolio_opportunity_id
    || metadata.opportunity_id
    || deepLinkParams.opportunityId
    || ''
  ).trim();
  return Boolean(String(event.source_id || '').trim() && opportunityId);
};

const canRelaunchTraceFollowUpEvent = (event: AgentDecisionTraceEvent) => {
  if (event.is_derived) return false;
  const eventKind = String(event.event_type || event.decision_type || '').trim().toLowerCase();
  if (!['follow_up_failed', 'follow_up_cancelled'].includes(eventKind)) return false;
  const afterState = (event.after_state && typeof event.after_state === 'object')
    ? (event.after_state as Record<string, any>)
    : {};
  const outcomeStatus = String(afterState.follow_up_outcome_status || event.status || '').trim().toLowerCase();
  const followUpJobId = String(afterState.follow_up_last_job_id || '').trim();
  return ['failed', 'cancelled'].includes(outcomeStatus) && Boolean(followUpJobId);
};

const isCompilerTraceEvent = (event: AgentDecisionTraceEvent) => (
  String(event.track_type || '').trim().toLowerCase() === 'compiler'
);

const scientificValidationStatusClasses = (status?: string | null) => {
  const normalized = String(status || '').trim().toLowerCase();
  if (normalized === 'blocked' || normalized === 'failed') return 'bg-rose-100 text-rose-700';
  if (normalized === 'running' || normalized === 'paused' || normalized === 'provisioning' || normalized === 'queued') return 'bg-amber-100 text-amber-800';
  if (normalized === 'succeeded' || normalized === 'completed') return 'bg-emerald-100 text-emerald-700';
  return 'bg-slate-100 text-slate-700';
};

const synthesisStatusClasses = (status?: string | null) => {
  const normalized = String(status || '').trim().toLowerCase();
  if (normalized === 'completed') return 'bg-emerald-100 text-emerald-700';
  if (normalized === 'failed' || normalized === 'cancelled') return 'bg-rose-100 text-rose-700';
  if (normalized) return 'bg-amber-100 text-amber-800';
  return 'bg-slate-100 text-slate-700';
};



const researchOpportunityStageClass = (value?: string | null) => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'completed') return 'bg-emerald-100 text-emerald-700';
  if (normalized === 'blocked' || normalized === 'suppressed') return 'bg-rose-100 text-rose-700';
  if (normalized === 'planned' || normalized === 'accepted') return 'bg-blue-100 text-blue-700';
  if (normalized === 'validating') return 'bg-amber-100 text-amber-800';
  return 'bg-slate-100 text-slate-700';
};

const formatOpportunityDelta = (nextValue: unknown, previousValue: unknown) => {
  const nextNum = Number(nextValue);
  const prevNum = Number(previousValue);
  if (!Number.isFinite(nextNum) || !Number.isFinite(prevNum)) return null;
  const delta = nextNum - prevNum;
  if (Math.abs(delta) < 0.0001) return '0.00';
  return `${delta > 0 ? '+' : ''}${delta.toFixed(2)}`;
};

const renderOpportunityReprioritizationMeta = (row: Record<string, any>) => {
  const reprioritizedAt = String(row.reprioritized_at || '').trim();
  if (!reprioritizedAt) return null;
  const confidenceDelta = formatOpportunityDelta(row.confidence, row.prior_confidence);
  const readinessDelta = formatOpportunityDelta(row.readiness, row.prior_readiness);
  const followUpReviewStatus = String(row.follow_up_review_status || '').trim();
  const childJobIds = Array.isArray(row.child_job_ids)
    ? row.child_job_ids.filter((value: unknown) => String(value || '').trim())
    : [];
  const sourceRunIds = Array.isArray(row.reprioritization_source_run_ids)
    ? row.reprioritization_source_run_ids.filter((value: unknown) => String(value || '').trim())
    : [];
  return (
    <div className="mt-2 rounded border border-emerald-200 bg-emerald-50 p-2 text-[11px] text-emerald-800">
      <div className="font-medium">Reprioritized from experiment evidence</div>
      <div className="mt-1">
        {new Date(reprioritizedAt).toLocaleString()}
        {row.reprioritization_reason ? ` · ${String(row.reprioritization_reason)}` : ''}
      </div>
      <div className="mt-1">
        Confidence {confidenceDelta ?? 'n/a'}
        {' '}· Readiness {readinessDelta ?? 'n/a'}
        {' '}· Autonomy {humanizeDecisionTraceValue(String(row.autonomy_state || 'eligible'))}
      </div>
      {followUpReviewStatus ? (
        <div className="mt-1">
          Follow-up {humanizeDecisionTraceValue(followUpReviewStatus)}
          {childJobIds.length > 0 ? ` · Child job ${childJobIds[0]}` : ''}
        </div>
      ) : null}
      {sourceRunIds.length > 0 ? (
        <div className="mt-1">Source runs {sourceRunIds.slice(0, 3).join(', ')}</div>
      ) : null}
    </div>
  );
};

const renderOpportunityFollowUpOutcomeMeta = (row: Record<string, any>) => {
  const outcomeStatus = String(row.follow_up_outcome_status || '').trim();
  if (!outcomeStatus) return null;
  const outcomeSummary = String(row.follow_up_outcome_summary || '').trim();
  const recordedAt = String(row.follow_up_outcome_recorded_at || '').trim();
  const childJobId = String(row.follow_up_last_job_id || (Array.isArray(row.child_job_ids) ? row.child_job_ids[0] : '') || '').trim();
  const badgeClass = outcomeStatus === 'completed'
    ? 'bg-emerald-100 text-emerald-700'
    : outcomeStatus === 'failed' || outcomeStatus === 'cancelled'
      ? 'bg-rose-100 text-rose-700'
      : 'bg-slate-100 text-slate-700';
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
      <span className={`px-2 py-0.5 rounded ${badgeClass}`}>
        Outcome {humanizeDecisionTraceValue(outcomeStatus)}
      </span>
      {recordedAt ? <span>{new Date(recordedAt).toLocaleString()}</span> : null}
      {childJobId ? <span>Job {childJobId}</span> : null}
      {outcomeSummary ? <span>{outcomeSummary}</span> : null}
    </div>
  );
};

const renderOpportunityReevaluationReviewMeta = (
  row: Record<string, any>,
  onNavigate?: (url: string) => void,
) => {
  const outcomeStatus = String(row.last_reevaluation_review_outcome || '').trim();
  if (!outcomeStatus) return null;
  const recordedAt = String(row.last_reevaluation_reviewed_at || '').trim();
  const reviewJobId = String(row.last_reevaluation_review_job_id || '').trim();
  const reviewNote = String(row.last_reevaluation_review_note || '').trim();
  const sourceNoteId = String(row.last_reevaluation_review_source_note_id || '').trim();
  const targetNoteId = String(row.last_reevaluation_review_target_note_id || '').trim();
  const openUrl = (url: string) => {
    if (!url || !onNavigate) return;
    onNavigate(url);
  };
  const badgeClass = outcomeStatus === 'dismissed'
    ? 'bg-slate-100 text-slate-700'
    : 'bg-violet-100 text-violet-700';
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
      <span className={`px-2 py-0.5 rounded ${badgeClass}`}>
        Reevaluation {humanizeDecisionTraceValue(outcomeStatus)}
      </span>
      {recordedAt ? <span>{new Date(recordedAt).toLocaleString()}</span> : null}
      {reviewNote ? <span>{reviewNote}</span> : null}
      {reviewJobId ? (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => openUrl(`/synthesis?job=${encodeURIComponent(reviewJobId)}`)}
        >
          Open reevaluation job
        </Button>
      ) : null}
      {sourceNoteId ? (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => openUrl(`/research-notes?note=${encodeURIComponent(sourceNoteId)}`)}
        >
          Open source note
        </Button>
      ) : null}
      {targetNoteId && targetNoteId !== sourceNoteId ? (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => openUrl(`/research-notes?note=${encodeURIComponent(targetNoteId)}`)}
        >
          Open saved note
        </Button>
      ) : null}
    </div>
  );
};

const canRelaunchOpportunityRow = (row: Record<string, any>) => {
  const outcomeStatus = String(row.follow_up_outcome_status || '').trim().toLowerCase();
  const lastJobId = String(row.follow_up_last_job_id || '').trim();
  return ['failed', 'cancelled'].includes(outcomeStatus) && Boolean(lastJobId);
};

type InboxHealthDrilldown = '' | 'completed_follow_up' | 'failed_follow_up' | 'cancelled_follow_up' | 'suppressed_relaunch';
type InboxPolicyDrilldown = '' | 'simulated_policy_impact' | 'policy_evaluation_after_rollout';
type QueueHealthDrilldown = '' | 'pending_follow_up_approvals' | 'manual_follow_up_recommendations' | 'blocked_follow_up';

const normalizeInboxHealthDrilldown = (value: unknown): InboxHealthDrilldown => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'completed_follow_up') return 'completed_follow_up';
  if (normalized === 'failed_follow_up') return 'failed_follow_up';
  if (normalized === 'cancelled_follow_up') return 'cancelled_follow_up';
  if (normalized === 'suppressed_relaunch') return 'suppressed_relaunch';
  return '';
};

const normalizeInboxPolicyDrilldown = (value: unknown): InboxPolicyDrilldown => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'simulated_policy_impact') return 'simulated_policy_impact';
  if (normalized === 'policy_evaluation_after_rollout') return 'policy_evaluation_after_rollout';
  return '';
};

const normalizeQueueHealthDrilldown = (value: unknown): QueueHealthDrilldown => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'pending_follow_up_approvals') return 'pending_follow_up_approvals';
  if (normalized === 'manual_follow_up_recommendations') return 'manual_follow_up_recommendations';
  if (normalized === 'blocked_follow_up') return 'blocked_follow_up';
  return '';
};

const formatInboxHealthDrilldownLabel = (value: InboxHealthDrilldown): string => {
  if (value === 'completed_follow_up') return 'completed outcomes';
  if (value === 'failed_follow_up') return 'failed outcomes';
  if (value === 'cancelled_follow_up') return 'cancelled outcomes';
  if (value === 'suppressed_relaunch') return 'suppressed relaunches';
  return '';
};

const formatInboxPolicyDrilldownLabel = (value: InboxPolicyDrilldown): string => {
  if (value === 'simulated_policy_impact') return 'simulated policy impact';
  if (value === 'policy_evaluation_after_rollout') return 'post-rollout evaluation';
  return '';
};

const formatQueueHealthDrilldownLabel = (value: QueueHealthDrilldown): string => {
  if (value === 'pending_follow_up_approvals') return 'pending approvals';
  if (value === 'manual_follow_up_recommendations') return 'manual recommendations';
  if (value === 'blocked_follow_up') return 'blocked follow-ups';
  return '';
};

const resolveOpportunityExplanationHeading = (row: Record<string, any>) => {
  const reviewStatus = String(row.follow_up_review_status || '').trim().toLowerCase();
  const autonomyState = String(row.autonomy_state || '').trim().toLowerCase();
  const stage = String(row.stage || '').trim().toLowerCase();
  if (reviewStatus === 'pending_approval') return 'Why queued';
  if (reviewStatus === 'rejected') return 'Why rejected';
  if (reviewStatus === 'manual_recommendation') return 'Why manual';
  if (autonomyState === 'blocked_structural' || stage === 'blocked') return 'Why blocked';
  if (autonomyState === 'cooldown') return 'Why cooling down';
  if (autonomyState === 'completed_waiting_change') return 'Why waiting';
  if (String(row.last_skip_reason_code || '').trim() || String(row.reason_code || '').trim()) return 'Why skipped';
  return 'Why this state';
};

const resolveOpportunityReasonCode = (row: Record<string, any>) => (
  String(
    row.last_decision_reason_code
    || row.last_blocked_reason_code
    || row.last_skip_reason_code
    || row.reason_code
    || ''
  ).trim()
);

const resolveOpportunityExplanationRows = (row: Record<string, any>) => {
  const supportingEvidence = Array.isArray(row.supporting_evidence)
    ? row.supporting_evidence.map((value: unknown) => String(value || '').trim()).filter(Boolean)
    : [];
  const childJobIds = Array.isArray(row.child_job_ids)
    ? row.child_job_ids.map((value: unknown) => String(value || '').trim()).filter(Boolean)
    : [];
  const sourceRunIds = Array.isArray(row.reprioritization_source_run_ids)
    ? row.reprioritization_source_run_ids.map((value: unknown) => String(value || '').trim()).filter(Boolean)
    : [];
  const reevaluationReviewOutcome = String(row.last_reevaluation_review_outcome || '').trim();
  const reevaluationReviewedAt = String(row.last_reevaluation_reviewed_at || '').trim();
  const reevaluationReviewJobId = String(row.last_reevaluation_review_job_id || '').trim();
  const reevaluationReviewNote = String(row.last_reevaluation_review_note || '').trim();
  const reevaluationReviewSourceNoteId = String(row.last_reevaluation_review_source_note_id || '').trim();
  const reevaluationReviewTargetNoteId = String(row.last_reevaluation_review_target_note_id || '').trim();
  const rows: Array<{ label: string; value: string }> = [];
  const reasonCode = resolveOpportunityReasonCode(row);
  const operatorNote = String(row.follow_up_review_note || row.operator_note || '').trim();
  const evidenceRevision = String(row.evidence_revision || '').trim();
  const nextEligibleAt = String(row.next_eligible_at || '').trim();
  const followUpReviewStatus = String(row.follow_up_review_status || '').trim();
  const autonomyState = String(row.autonomy_state || '').trim();
  const stage = String(row.stage || '').trim();
  const hypothesis = String(row.hypothesis || '').trim();
  const sourceRuns = sourceRunIds.slice(0, 3).join(', ');
  const childJobs = childJobIds.slice(0, 3).join(', ');
  const confidenceDelta = formatOpportunityDelta(row.confidence, row.prior_confidence);
  const readinessDelta = formatOpportunityDelta(row.readiness, row.prior_readiness);
  const followUpOutcomeStatus = String(row.follow_up_outcome_status || '').trim();
  const followUpOutcomeSummary = String(row.follow_up_outcome_summary || '').trim();
  const followUpOutcomeRecordedAt = String(row.follow_up_outcome_recorded_at || '').trim();
  const followUpLastJobId = String(row.follow_up_last_job_id || '').trim();

  if (reasonCode) rows.push({ label: 'Reason', value: humanizeDecisionTraceValue(reasonCode) });
  if (operatorNote) rows.push({ label: 'Note', value: operatorNote });
  if (supportingEvidence.length > 0) rows.push({ label: 'Evidence', value: supportingEvidence.slice(0, 3).join(' · ') });
  else if (hypothesis) rows.push({ label: 'Hypothesis', value: hypothesis });
  if (followUpReviewStatus) rows.push({ label: 'Review', value: humanizeDecisionTraceValue(followUpReviewStatus) });
  if (autonomyState || stage) rows.push({ label: 'State', value: [autonomyState && humanizeDecisionTraceValue(autonomyState), stage && humanizeDecisionTraceValue(stage)].filter(Boolean).join(' · ') });
  if (evidenceRevision) rows.push({ label: 'Evidence rev', value: evidenceRevision });
  if (nextEligibleAt) rows.push({ label: 'Next eligible', value: new Date(nextEligibleAt).toLocaleString() });
  if (sourceRuns) rows.push({ label: 'Source runs', value: sourceRuns });
  if (childJobs) rows.push({ label: 'Child jobs', value: childJobs });
  if (followUpOutcomeStatus) rows.push({ label: 'Outcome', value: humanizeDecisionTraceValue(followUpOutcomeStatus) });
  if (followUpOutcomeRecordedAt) rows.push({ label: 'Outcome at', value: new Date(followUpOutcomeRecordedAt).toLocaleString() });
  if (followUpLastJobId) rows.push({ label: 'Outcome job', value: followUpLastJobId });
  if (followUpOutcomeSummary) rows.push({ label: 'Outcome summary', value: followUpOutcomeSummary });
  if (reevaluationReviewOutcome) rows.push({ label: 'Reevaluation review', value: humanizeDecisionTraceValue(reevaluationReviewOutcome) });
  if (reevaluationReviewedAt) rows.push({ label: 'Reevaluation at', value: new Date(reevaluationReviewedAt).toLocaleString() });
  if (reevaluationReviewJobId) rows.push({ label: 'Reevaluation job', value: reevaluationReviewJobId });
  if (reevaluationReviewSourceNoteId) rows.push({ label: 'Reevaluation source note', value: reevaluationReviewSourceNoteId });
  if (reevaluationReviewTargetNoteId && reevaluationReviewTargetNoteId !== reevaluationReviewSourceNoteId) rows.push({ label: 'Reevaluation saved note', value: reevaluationReviewTargetNoteId });
  if (reevaluationReviewNote) rows.push({ label: 'Reevaluation note', value: reevaluationReviewNote });
  if (confidenceDelta || readinessDelta) rows.push({ label: 'Score delta', value: `Confidence ${confidenceDelta ?? 'n/a'} · Readiness ${readinessDelta ?? 'n/a'}` });
  return rows;
};

const codingSwarmPresetLabel = (presetKey?: string | null) => {
  const normalized = String(presetKey || '').trim().toLowerCase();
  if (normalized === 'build_break_swarm') return 'Build Break Swarm';
  if (normalized === 'frontend_regression_swarm') return 'Frontend Regression Swarm';
  return 'Bug Triage Swarm';
};

const buildScientificSandboxProfileDraft = (profile?: ScientificSandboxProfile | null) => ({
  id: String(profile?.id || ''),
  name: String(profile?.name || ''),
  description: String(profile?.description || ''),
  track_type: String(profile?.track_type || 'generic'),
  backend: String(profile?.backend || 'docker'),
  docker_image: String(profile?.docker_image || 'python:3.11-slim'),
  timeout_seconds: String(profile?.timeout_seconds ?? 900),
  memory_mb: String((profile?.resource_caps as any)?.memory_mb ?? 2048),
  cpus: String((profile?.resource_caps as any)?.cpus ?? 1.5),
  pids_limit: String((profile?.resource_caps as any)?.pids_limit ?? 192),
  allowed_benchmark_families: Array.isArray(profile?.allowed_benchmark_families) ? profile!.allowed_benchmark_families.join('\n') : 'generic_validation',
  allowed_perf_collectors: Array.isArray(profile?.allowed_perf_collectors) ? profile!.allowed_perf_collectors.join('\n') : 'benchmark_output',
  required_capabilities: Array.isArray(profile?.required_capabilities) ? profile!.required_capabilities.join('\n') : 'repo_reconstruction',
  toolchains: Array.isArray(profile?.toolchains) ? profile!.toolchains.join('\n') : 'python\npytest',
  budget_limit_default: String(profile?.budget_limit_default ?? 25),
  enabled: Boolean(profile?.enabled ?? true),
  is_default: Boolean(profile?.is_default ?? false),
});

const AutonomousAgentsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'queue' | 'trace' | 'health' | 'jobs' | 'swarm' | 'outcomes' | 'profiles' | 'templates' | 'chains' | 'inbox' | 'backlog' | 'domain' | 'fleet' | 'create'>('jobs');
  const [showSystemMap, setShowSystemMap] = useState(false);
  const [selectedJob, setSelectedJob] = useState<AgentJob | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [launchModeFilter, setLaunchModeFilter] = useState<string>('');
  const [hasRelaunchChildrenFilter, setHasRelaunchChildrenFilter] = useState<string>('');
  const [relaunchFromJobIdFilter, setRelaunchFromJobIdFilter] = useState<string>('');
  const [swarmOnlyFilter, setSwarmOnlyFilter] = useState<boolean>(false);
  const [swarmSortBy, setSwarmSortBy] = useState<string>('created_desc');
  const [swarmMinConsensus, setSwarmMinConsensus] = useState<number>(0);
  const [graphHealthFilter, setGraphHealthFilter] = useState<string>('');
  const [graphSortBy, setGraphSortBy] = useState<string>('none');
  const [dedupSkipFilter, setDedupSkipFilter] = useState<string>('');
  const [scopeGuardFilter, setScopeGuardFilter] = useState<string>('');
  const [experimentRecoveryFilter, setExperimentRecoveryFilter] = useState<string>('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showCustomerResearchModal, setShowCustomerResearchModal] = useState(false);
  const [showInboxMonitorModal, setShowInboxMonitorModal] = useState(false);
  const [showMonitorProfilesModal, setShowMonitorProfilesModal] = useState(false);
  const [createFromTemplate, setCreateFromTemplate] = useState<AgentJobTemplate | null>(null);
  const [startFromChain, setStartFromChain] = useState<AgentJobChainDefinition | null>(null);
  const [showClaudeQuickStartModal, setShowClaudeQuickStartModal] = useState(false);
  const [showDomainResearchQuickStartModal, setShowDomainResearchQuickStartModal] = useState(false);
  const [showRepoBugTriageQuickStartModal, setShowRepoBugTriageQuickStartModal] = useState(false);
  const [showBugTriageSwarmQuickStartModal, setShowBugTriageSwarmQuickStartModal] = useState(false);
  const [showBuildBreakSwarmQuickStartModal, setShowBuildBreakSwarmQuickStartModal] = useState(false);
  const [showFrontendRegressionSwarmQuickStartModal, setShowFrontendRegressionSwarmQuickStartModal] = useState(false);
  const [codingSwarmLaunchSeed, setCodingSwarmLaunchSeed] = useState<{ presetKey: string; profileId?: string; sourceId?: string } | null>(null);
  const [showRoleWorkflowQuickStartModal, setShowRoleWorkflowQuickStartModal] = useState(false);
  const [swarmReviewPresetFilter, setSwarmReviewPresetFilter] = useState<string>('');
  const [swarmReviewStateFilter, setSwarmReviewStateFilter] = useState<string>('');
  const [swarmReviewConfidenceBand, setSwarmReviewConfidenceBand] = useState<string>('');
  const [swarmReviewBacklogFilter, setSwarmReviewBacklogFilter] = useState<string>('');
  const [swarmReviewVisibilityScope, setSwarmReviewVisibilityScope] = useState<'mine' | 'shared' | 'all'>('mine');
  const [swarmReviewAssignmentFilter, setSwarmReviewAssignmentFilter] = useState<string>('');
  const [swarmOutcomePresetFilter, setSwarmOutcomePresetFilter] = useState<string>('');
  const [swarmOutcomeTerminalFilter, setSwarmOutcomeTerminalFilter] = useState<string>('');
  const [swarmOutcomePromotionFilter, setSwarmOutcomePromotionFilter] = useState<string>('');
  const [swarmOutcomeDateRange, setSwarmOutcomeDateRange] = useState<string>('all');
  const [swarmOutcomeVisibilityScope, setSwarmOutcomeVisibilityScope] = useState<'mine' | 'shared' | 'all'>('mine');
  const [profilePresetFilter, setProfilePresetFilter] = useState<string>('');
  const [profileSourceFilter, setProfileSourceFilter] = useState<string>('');
  const [profileStatusFilter, setProfileStatusFilter] = useState<string>('');
  const [profileDefaultOnly, setProfileDefaultOnly] = useState<boolean>(false);
  const [profileVisibilityFilter, setProfileVisibilityFilter] = useState<string>('');
  const [profileOwnershipFilter, setProfileOwnershipFilter] = useState<string>('');
  const [profileOwnerFilter, setProfileOwnerFilter] = useState<string>('');
  const [editingCodingSwarmProfileId, setEditingCodingSwarmProfileId] = useState<string>('');
  const [codingSwarmProfileDraft, setCodingSwarmProfileDraft] = useState<CodingSwarmProfileUpdate & { source_id?: string; duplicate_mode?: boolean; title: string }>({
    title: '',
    source_id: '',
    preset_key: 'bug_triage_swarm',
    description: '',
    scope_default: 'auto',
    default_commands: [],
    default_file_paths: [],
    max_agents: 4,
    safe_command_policy: 'standard',
    saved_search_query: '',
    is_default: false,
    status: 'active',
    visibility: 'private',
    shared_with_user_ids: [],
    profile_metadata: {},
    duplicate_mode: false,
  });
  const [templateRecommendScope, setTemplateRecommendScope] = useState<string>('');
  const [templateRecommendGoal, setTemplateRecommendGoal] = useState<string>('');
  const [traceSourceKindFilter, setTraceSourceKindFilter] = useState<string>('');
  const [traceDecisionTypeFilter, setTraceDecisionTypeFilter] = useState<string>('');
  const [traceCustomerFilter, setTraceCustomerFilter] = useState<string>('');
  const [traceStatusFilter, setTraceStatusFilter] = useState<string>('');
  const [traceSeverityFilter, setTraceSeverityFilter] = useState<string>('');
  const [traceActorModeFilter, setTraceActorModeFilter] = useState<string>('');
  const [traceTriageStatusFilter, setTraceTriageStatusFilter] = useState<string>('');
  const [traceAssignedToUserIdFilter, setTraceAssignedToUserIdFilter] = useState<string>('');
  const [traceUnassignedOnly, setTraceUnassignedOnly] = useState<boolean>(false);
  const [traceEscalationStateFilter, setTraceEscalationStateFilter] = useState<string>('');
  const [tracePinnedOnly, setTracePinnedOnly] = useState<boolean>(false);
  const [traceActionableOnly, setTraceActionableOnly] = useState<boolean>(false);
  const [traceDateRange, setTraceDateRange] = useState<string>('7d');
  const [traceOffset, setTraceOffset] = useState<number>(0);
  const [traceOperatorPreset, setTraceOperatorPreset] = useState<string>('');
  const [expandedTraceEventId, setExpandedTraceEventId] = useState<string>('');
  const [traceActionNoteDrafts, setTraceActionNoteDrafts] = useState<Record<string, string>>({});
  const [traceAssigneeDrafts, setTraceAssigneeDrafts] = useState<Record<string, string>>({});
  const [traceDueAtDrafts, setTraceDueAtDrafts] = useState<Record<string, string>>({});
  const [selectedTraceViewId, setSelectedTraceViewId] = useState<string>('');
  const [traceViewNameDraft, setTraceViewNameDraft] = useState<string>('');
  const [traceViewIsDefaultDraft, setTraceViewIsDefaultDraft] = useState<boolean>(false);
  const [selectedChainStatus, setSelectedChainStatus] = useState<AgentJobChainStatus | null>(null);
  const [showExportModal, setShowExportModal] = useState(false);
  const [backlogTitle, setBacklogTitle] = useState('');
  const [backlogGoal, setBacklogGoal] = useState('');
  const [backlogSourceId, setBacklogSourceId] = useState('');
  const [backlogScope, setBacklogScope] = useState('auto');
  const [backlogFailureSymptom, setBacklogFailureSymptom] = useState('');
  const [backlogCommandsText, setBacklogCommandsText] = useState('');
  const [backlogFilePathsText, setBacklogFilePathsText] = useState('');
  const [backlogVisibilityScope, setBacklogVisibilityScope] = useState<'mine' | 'shared' | 'all'>('mine');
  const [backlogAssignmentFilter, setBacklogAssignmentFilter] = useState<string>('');
  const [backlogQueueStateFilter, setBacklogQueueStateFilter] = useState<string>('');
  const [backlogNoteDrafts, setBacklogNoteDrafts] = useState<Record<string, string>>({});
  const [backlogCloseReasonDrafts, setBacklogCloseReasonDrafts] = useState<Record<string, string>>({});
  const [swarmReviewNoteDrafts, setSwarmReviewNoteDrafts] = useState<Record<string, string>>({});
  const [domainProfileTitle, setDomainProfileTitle] = useState('');
  const [domainProfileTopic, setDomainProfileTopic] = useState('');
  const [domainProfileObjective, setDomainProfileObjective] = useState('');
  const [domainProfileTrackType, setDomainProfileTrackType] = useState<'compiler' | 'microarchitecture' | 'generic'>('compiler');
  const [domainProfileSourceScope, setDomainProfileSourceScope] = useState<'kb_only' | 'arxiv_only' | 'kb_plus_arxiv' | 'kb_plus_arxiv_plus_repo'>('kb_plus_arxiv_plus_repo');
  const [domainProfileQueriesText, setDomainProfileQueriesText] = useState('');
  const [domainProfileBenchmarkQueriesText, setDomainProfileBenchmarkQueriesText] = useState('');
  const [domainProfileCadenceMinutes, setDomainProfileCadenceMinutes] = useState('1440');
  const [domainProfileRepoSelection, setDomainProfileRepoSelection] = useState<Record<string, boolean>>({});
  const [domainProfileSandboxProfileId, setDomainProfileSandboxProfileId] = useState('');
  const [domainProfilePolicyDrafts, setDomainProfilePolicyDrafts] = useState<Record<string, DomainResearchProfilePolicyDraft>>({});
  const [portfolioTitle, setPortfolioTitle] = useState('');
  const [portfolioObjective, setPortfolioObjective] = useState('');
  const [portfolioProfileSelection, setPortfolioProfileSelection] = useState<Record<string, boolean>>({});
  const [portfolioSandboxProfileId, setPortfolioSandboxProfileId] = useState('');
  const [portfolioPolicyDrafts, setPortfolioPolicyDrafts] = useState<Record<string, ResearchPortfolioPolicyDraft>>({});
  const [expandedPortfolioIds, setExpandedPortfolioIds] = useState<Record<string, boolean>>({});
  const [expandedDomainProfileIds, setExpandedDomainProfileIds] = useState<Record<string, boolean>>({});
  const [highlightedAutonomyRowKey, setHighlightedAutonomyRowKey] = useState<string>('');
  const [highlightedAutonomyCardKey, setHighlightedAutonomyCardKey] = useState<string>('');
  const [expandedOpportunityExplanationRows, setExpandedOpportunityExplanationRows] = useState<Record<string, boolean>>({});
  const [followUpReviewNoteDrafts, setFollowUpReviewNoteDrafts] = useState<Record<string, string>>({});
  const [activeFollowUpReviewKey, setActiveFollowUpReviewKey] = useState<string>('');
  const [bulkFollowUpSelection, setBulkFollowUpSelection] = useState<Record<string, boolean>>({});
  const [bulkFollowUpNotes, setBulkFollowUpNotes] = useState<Record<string, string>>({});
  const [activeBulkFollowUpOwnerKey, setActiveBulkFollowUpOwnerKey] = useState<string>('');
  const [opportunityNoteDraft, setOpportunityNoteDraft] = useState<{
    mode: 'suppress' | 'launch' | 'relaunch';
    surface: 'domain' | 'fleet';
    ownerId: string;
    opportunityId: string;
    value: string;
  } | null>(null);
  const [showDisabledSandboxProfiles, setShowDisabledSandboxProfiles] = useState(false);
  const [editingScientificSandboxProfileId, setEditingScientificSandboxProfileId] = useState('');
  const [sandboxProfileDraft, setSandboxProfileDraft] = useState(() => buildScientificSandboxProfileDraft());
  const [exportingJob, setExportingJob] = useState<AgentJob | null>(null);
  const landingTabInitializedRef = useRef(false);

  const [inboxStatusFilter, setInboxStatusFilter] = useState<ResearchInboxItemStatus | ''>('');
  const [inboxTypeFilter, setInboxTypeFilter] = useState<string>('');
  const [inboxSearch, setInboxSearch] = useState<string>('');
  const [inboxCustomerFilter, setInboxCustomerFilter] = useState<string>('');
  const [inboxJobFilter, setInboxJobFilter] = useState<string>('');
  const [inboxHealthDrilldown, setInboxHealthDrilldown] = useState<InboxHealthDrilldown>('');
  const [inboxPolicyDrilldown, setInboxPolicyDrilldown] = useState<InboxPolicyDrilldown>('');
  const [selectedInboxIds, setSelectedInboxIds] = useState<Record<string, boolean>>({});
  const [inboxResearchGoalDraft, setInboxResearchGoalDraft] = useState<string>(
    'Deep-dive on the selected Research Inbox items and propose concrete next steps (hypotheses + experiment plan).'
  );
  const [inboxBulkRejectReason, setInboxBulkRejectReason] = useState<string>('');
  const [inboxBulkFollowUpNote, setInboxBulkFollowUpNote] = useState<string>('');
  const [inboxRejectReasonDrafts, setInboxRejectReasonDrafts] = useState<Record<string, string>>({});
  const [inboxMuteTokenDrafts, setInboxMuteTokenDrafts] = useState<Record<string, string>>({});
  const [inboxMutePhraseDrafts, setInboxMutePhraseDrafts] = useState<Record<string, string>>({});
  const [paperRepoSelectionDrafts, setPaperRepoSelectionDrafts] = useState<Record<string, string>>({});
  const [queueItemTypeFilter, setQueueItemTypeFilter] = useState<string>('');
  const [queueStatusFilter, setQueueStatusFilter] = useState<string>('');
  const [queueCustomerFilter, setQueueCustomerFilter] = useState<string>('');
  const [queueJobFilter, setQueueJobFilter] = useState<string>('');
  const [queueHealthDrilldown, setQueueHealthDrilldown] = useState<QueueHealthDrilldown>('');
  const [queueJobTypeFilter, setQueueJobTypeFilter] = useState<string>('');
  const [queueSlaBucketFilter, setQueueSlaBucketFilter] = useState<string>('');
  const [queueEscalationFilter, setQueueEscalationFilter] = useState<string>('');
  const [queueOverdueOnly, setQueueOverdueOnly] = useState<boolean>(false);
  const [queueSortBy, setQueueSortBy] = useState<string>('priority_score_desc');
  const [queueOperatorPreset, setQueueOperatorPreset] = useState<string>('');
  const [queueDrafts, setQueueDrafts] = useState<Record<string, {
    note: string;
    showEdit: boolean;
    tool: string;
    purpose: string;
    params: string;
  }>>({});
  const [queueSelection, setQueueSelection] = useState<Record<string, boolean>>({});
  const [queueBulkNote, setQueueBulkNote] = useState<string>('');
  const [healthCustomerFilter, setHealthCustomerFilter] = useState<string>('');
  const healthMonitorCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const autonomyTargetRowRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const autonomyTargetCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [healthMonitorTypeFilter, setHealthMonitorTypeFilter] = useState<string>('');
  const [healthBucketFilter, setHealthBucketFilter] = useState<string>('');
  const [healthAutonomyFilter, setHealthAutonomyFilter] = useState<string>('');
  const [healthPolicyDrafts, setHealthPolicyDrafts] = useState<Record<string, { automation_profile: string; mode: string; allowed: string[] }>>({});
  const [healthBudgetDrafts, setHealthBudgetDrafts] = useState<Record<string, { auto_launch_limit_24h: number; approval_queue_limit_24h: number; alert_limit_24h: number; queue_backlog_cap: number }>>({});
  const [healthCustomerBudgetDrafts, setHealthCustomerBudgetDrafts] = useState<Record<string, { auto_launch_limit_24h: number; approval_queue_limit_24h: number; alert_limit_24h: number; queue_backlog_cap: number }>>({});
  const [healthPolicySimulations, setHealthPolicySimulations] = useState<Record<string, ResearchMonitorPolicySimulationResponse>>({});
  const [healthCustomerRebalancePreviews, setHealthCustomerRebalancePreviews] = useState<Record<string, ResearchMonitorCustomerRebalancePreview>>({});
  const [healthPolicyEvaluations, setHealthPolicyEvaluations] = useState<Record<string, ResearchMonitorPolicyEvaluationDetail>>({});
  const [healthCustomerRebalanceEvaluations, setHealthCustomerRebalanceEvaluations] = useState<Record<string, ResearchMonitorCustomerRebalanceEvaluationDetail>>({});

  const queryClient = useQueryClient();
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin';
  const traceFiltersDirtyRef = useRef(false);
  const deepLinkedJobId = useMemo(() => new URLSearchParams(location.search).get('job'), [location.search]);
  const deepLinkedTraceTab = useMemo(() => String(new URLSearchParams(location.search).get('tab') || '').trim().toLowerCase() === 'trace', [location.search]);
  const deepLinkedTraceEventId = useMemo(() => String(new URLSearchParams(location.search).get('trace_event') || '').trim(), [location.search]);
  const deepLinkedHealthTab = useMemo(() => String(new URLSearchParams(location.search).get('tab') || '').trim().toLowerCase() === 'health', [location.search]);
  const deepLinkedQueueTab = useMemo(() => String(new URLSearchParams(location.search).get('tab') || '').trim().toLowerCase() === 'queue', [location.search]);
  const deepLinkedQueueCustomer = useMemo(() => new URLSearchParams(location.search).get('queue_customer'), [location.search]);
  const deepLinkedQueueJobId = useMemo(() => new URLSearchParams(location.search).get('queue_job'), [location.search]);
  const deepLinkedQueueHealthDrilldown = useMemo(
    () => normalizeQueueHealthDrilldown(new URLSearchParams(location.search).get('queue_health_drilldown')),
    [location.search]
  );
  const deepLinkedDomainTab = useMemo(() => String(new URLSearchParams(location.search).get('tab') || '').trim().toLowerCase() === 'domain', [location.search]);
  const deepLinkedFleetTab = useMemo(() => String(new URLSearchParams(location.search).get('tab') || '').trim().toLowerCase() === 'fleet', [location.search]);
  const deepLinkedInboxTab = useMemo(() => String(new URLSearchParams(location.search).get('tab') || '').trim().toLowerCase() === 'inbox', [location.search]);
  const deepLinkedInboxId = useMemo(() => new URLSearchParams(location.search).get('inbox'), [location.search]);
  const deepLinkedInboxJobId = useMemo(() => new URLSearchParams(location.search).get('inbox_job'), [location.search]);
  const deepLinkedInboxCustomer = useMemo(() => new URLSearchParams(location.search).get('inbox_customer'), [location.search]);
  const deepLinkedInboxHealthDrilldown = useMemo(
    () => normalizeInboxHealthDrilldown(new URLSearchParams(location.search).get('inbox_health_drilldown')),
    [location.search]
  );
  const deepLinkedInboxPolicyDrilldown = useMemo(
    () => normalizeInboxPolicyDrilldown(new URLSearchParams(location.search).get('inbox_policy_drilldown')),
    [location.search]
  );
  const deepLinkedHealthCustomer = useMemo(() => new URLSearchParams(location.search).get('health_customer'), [location.search]);
  const deepLinkedHealthMonitor = useMemo(() => new URLSearchParams(location.search).get('health_monitor'), [location.search]);
  const deepLinkedHealthPolicyHistory = useMemo(() => new URLSearchParams(location.search).get('health_policy_history'), [location.search]);
  const deepLinkedFleetId = useMemo(() => String(new URLSearchParams(location.search).get('fleetId') || '').trim(), [location.search]);
  const deepLinkedProfileId = useMemo(() => String(new URLSearchParams(location.search).get('profileId') || '').trim(), [location.search]);
  const deepLinkedOpportunityId = useMemo(() => String(new URLSearchParams(location.search).get('opportunityId') || '').trim(), [location.search]);
  const isRelaunchFromJobIdFilterValid = useMemo(() => {
    const v = String(relaunchFromJobIdFilter || '').trim();
    if (!v) return true;
    return /^[0-9a-fA-F-]{36}$/.test(v);
  }, [relaunchFromJobIdFilter]);
  const hasExplicitTraceFilterParams = useMemo(
    () => {
      const params = new URLSearchParams(location.search);
      return TRACE_FILTER_QUERY_KEYS.some((key) => {
        if (!params.has(key)) return false;
        if (key === 'trace_date_range') {
          const value = String(params.get(key) || '').trim() || '7d';
          return value !== '7d';
        }
        return true;
      });
    },
    [location.search]
  );
  const hasExplicitTraceContext = useMemo(
    () => hasExplicitTraceFilterParams || Boolean(deepLinkedTraceEventId),
    [deepLinkedTraceEventId, hasExplicitTraceFilterParams]
  );

  const buildAutonomousAgentsUrl = useCallback(
    (jobId?: string, extras?: Record<string, string | null | undefined>) => {
      const params = new URLSearchParams(location.search);
      if (jobId && String(jobId).trim()) {
        params.set('job', String(jobId).trim());
      } else {
        params.delete('job');
      }
      Object.entries(extras || {}).forEach(([key, value]) => {
        const text = String(value || '').trim();
        if (text) {
          params.set(key, text);
        } else {
          params.delete(key);
        }
      });
      const qs = params.toString();
      return `${location.pathname}${qs ? `?${qs}` : ''}`;
    },
    [location.pathname, location.search]
  );

  const buildResearchNoteExperimentUrl = useCallback(
    (noteId?: string | null, extras?: Record<string, string | null | undefined>) => {
      const params = new URLSearchParams();
      const normalizedNoteId = String(noteId || '').trim();
      if (normalizedNoteId) {
        params.set('note', normalizedNoteId);
      }
      Object.entries(extras || {}).forEach(([key, value]) => {
        const text = String(value || '').trim();
        if (text) {
          params.set(key, text);
        }
      });
      const qs = params.toString();
      return `/research-notes${qs ? `?${qs}` : ''}`;
    },
    []
  );

  const openBacklogJob = useCallback((jobId: string) => {
    const normalizedJobId = String(jobId || '').trim();
    if (!normalizedJobId) return;
    setActiveTab('jobs');
    navigate(buildAutonomousAgentsUrl(normalizedJobId), { replace: true });
  }, [navigate, buildAutonomousAgentsUrl]);

  const openPatchPr = useCallback((patchPrId: string) => {
    const normalizedPatchPrId = String(patchPrId || '').trim();
    if (!normalizedPatchPrId) return;
    navigate(`/patch-prs?pr=${encodeURIComponent(normalizedPatchPrId)}`);
  }, [navigate]);

  const registerHealthMonitorCardRef = useCallback((monitorJobId: string) => (node: HTMLDivElement | null) => {
    if (!monitorJobId) return;
    healthMonitorCardRefs.current[monitorJobId] = node;
  }, []);

  const openHealthMonitorFocus = useCallback((customer: string, monitorJobId: string) => {
    const normalizedCustomer = String(customer || '').trim();
    const normalizedMonitorJobId = String(monitorJobId || '').trim();
    setActiveTab('health');
    setHealthCustomerFilter(normalizedCustomer);
    navigate(buildAutonomousAgentsUrl(undefined, {
      tab: 'health',
      health_customer: normalizedCustomer || null,
      health_monitor: normalizedMonitorJobId || null,
    }), { replace: true });
  }, [buildAutonomousAgentsUrl, navigate]);

  const openInboxHealthDrilldown = useCallback((
    drilldown: InboxHealthDrilldown,
    context?: { customer?: string | null; monitorJobId?: string | null }
  ) => {
    const customer = String(context?.customer || '').trim();
    const monitorJobId = String(context?.monitorJobId || '').trim();
    setActiveTab('inbox');
    setInboxStatusFilter('accepted');
    setInboxTypeFilter('');
    setInboxSearch('');
    setInboxCustomerFilter(customer);
    setInboxJobFilter(monitorJobId);
    setInboxHealthDrilldown(drilldown);
    setInboxPolicyDrilldown('');
    navigate(buildAutonomousAgentsUrl(undefined, {
      tab: 'inbox',
      inbox_customer: customer || null,
      inbox_job: monitorJobId || null,
      inbox_health_drilldown: drilldown || null,
      inbox_policy_drilldown: null,
    }), { replace: true });
  }, [buildAutonomousAgentsUrl, navigate]);

  const openQueueHealthDrilldown = useCallback((
    drilldown: QueueHealthDrilldown,
    context?: { customer?: string | null; monitorJobId?: string | null }
  ) => {
    const customer = String(context?.customer || '').trim();
    const monitorJobId = String(context?.monitorJobId || '').trim();
    setActiveTab('queue');
    setQueueJobTypeFilter('');
    setQueueSlaBucketFilter('');
    setQueueEscalationFilter('');
    setQueueOverdueOnly(false);
    setQueueSortBy('priority_score_desc');
    navigate(buildAutonomousAgentsUrl(undefined, {
      tab: 'queue',
      queue_item_type: 'follow_up_recommendation',
      queue_customer: customer || null,
      queue_job: monitorJobId || null,
      queue_health_drilldown: drilldown || null,
    }), { replace: true });
  }, [buildAutonomousAgentsUrl, navigate]);

  const openDecisionTraceTarget = useCallback((event: AgentDecisionTraceEvent) => {
    const deepLink = event.deep_link;
    if (!deepLink) return;
    const params: Record<string, string | null | undefined> = {
      tab: String(deepLink.target_tab || '').trim() || undefined,
      ...(deepLink.params || {}),
    };
    const nextJobId = String(deepLink.job_id || params.job || '').trim() || undefined;
    if (deepLink.target_tab === 'trace') {
      setActiveTab('trace');
    } else if (deepLink.target_tab === 'queue') {
      setActiveTab('queue');
    } else if (deepLink.target_tab === 'health') {
      setActiveTab('health');
    } else if (deepLink.target_tab === 'domain') {
      setActiveTab('domain');
    } else if (deepLink.target_tab === 'fleet') {
      setActiveTab('fleet');
    } else if (deepLink.target_tab === 'inbox') {
      setActiveTab('inbox');
    } else if (deepLink.target_tab === 'jobs') {
      setActiveTab('jobs');
    }
    navigate(buildAutonomousAgentsUrl(nextJobId, params), { replace: true });
  }, [buildAutonomousAgentsUrl, navigate]);

  const openDecisionTraceResearchNote = useCallback((noteId?: string | null) => {
    const normalized = String(noteId || '').trim();
    if (!normalized) return;
    navigate(`/research-notes?note=${encodeURIComponent(normalized)}`);
  }, [navigate]);

  const openDecisionTraceReevaluationJob = useCallback((jobId?: string | null) => {
    const normalized = String(jobId || '').trim();
    if (!normalized) return;
    navigate(`/synthesis?job=${encodeURIComponent(normalized)}`);
  }, [navigate]);

  const buildAutonomyCardKey = useCallback((scope: 'domain' | 'fleet', ownerId: string) => (
    `${scope}:${String(ownerId || '').trim()}`
  ), []);

  const buildAutonomyOpportunityRowKey = useCallback((scope: 'domain' | 'fleet', ownerId: string, opportunityId: string) => (
    `${scope}:${String(ownerId || '').trim()}:opportunity:${String(opportunityId || '').trim()}`
  ), []);

  const buildAutonomyReviewRowKey = useCallback((
    scope: 'domain' | 'fleet',
    ownerId: string,
    reviewKind: 'pending' | 'manual' | 'suppressed',
    opportunityId: string,
  ) => (
    `${scope}:${String(ownerId || '').trim()}:review:${reviewKind}:${String(opportunityId || '').trim()}`
  ), []);

  const registerAutonomyCardRef = useCallback((key: string) => (node: HTMLDivElement | null) => {
    if (!key) return;
    autonomyTargetCardRefs.current[key] = node;
  }, []);

  const registerAutonomyRowRef = useCallback((key: string) => (node: HTMLDivElement | null) => {
    if (!key) return;
    autonomyTargetRowRefs.current[key] = node;
  }, []);

  const renderAutonomySummaryRow = useCallback((
    scope: 'domain' | 'fleet',
    ownerId: string,
    reviewKind: 'pending' | 'manual' | 'suppressed',
    row: Record<string, any>,
    idx: number,
    content: React.ReactNode,
  ) => {
    const opportunityId = String(row.opportunity_id || row.canonical_key || idx).trim();
    const rowKey = buildAutonomyReviewRowKey(scope, ownerId, reviewKind, opportunityId);
    return (
      <div
        key={rowKey}
        ref={registerAutonomyRowRef(rowKey)}
        className={`rounded border px-2 py-1 text-gray-600 transition-colors ${highlightedAutonomyRowKey === rowKey ? AUTONOMY_FOCUS_ROW_CLASS : 'border-transparent'}`}
      >
        {content}
      </div>
    );
  }, [buildAutonomyReviewRowKey, highlightedAutonomyRowKey, registerAutonomyRowRef]);

  const resolveOpportunityContextRow = useCallback((
    row: Record<string, any>,
    opportunities: Array<Record<string, any>> | undefined | null,
  ) => {
    const opportunityId = String(row.opportunity_id || '').trim();
    if (!opportunityId || !Array.isArray(opportunities)) return row;
    return opportunities.find((candidate) => String(candidate?.opportunity_id || '').trim() === opportunityId) || row;
  }, []);

  const downloadBacklogProposal = useCallback(async (proposalId: string, title?: string | null) => {
    const normalizedProposalId = String(proposalId || '').trim();
    if (!normalizedProposalId) return;
    try {
      await apiClient.downloadCodePatchProposal(normalizedProposalId, String(title || `proposal-${normalizedProposalId}`));
    } catch (error: any) {
      toast.error(error?.message || 'Failed to download proposal');
    }
  }, []);

  // Fetch jobs
  const { data: jobsData, isLoading: jobsLoading, refetch: refetchJobs } = useQuery(
    ['agent-jobs', statusFilter, typeFilter, launchModeFilter, hasRelaunchChildrenFilter, relaunchFromJobIdFilter, swarmOnlyFilter, swarmSortBy, swarmMinConsensus],
    () => apiClient.listAgentJobs({
      status: statusFilter || undefined,
      job_type: typeFilter || undefined,
      launch_mode: launchModeFilter || undefined,
      relaunch_from_job_id: /^[0-9a-fA-F-]{36}$/.test(String(relaunchFromJobIdFilter || ''))
        ? relaunchFromJobIdFilter
        : undefined,
      has_relaunch_children:
        hasRelaunchChildrenFilter === 'yes'
          ? true
          : hasRelaunchChildrenFilter === 'no'
            ? false
            : undefined,
      swarm_only: swarmOnlyFilter || undefined,
      swarm_min_consensus: swarmMinConsensus > 0 ? swarmMinConsensus : undefined,
      sort_by: swarmSortBy || undefined,
      page_size: 50,
    }),
    {
      refetchInterval: 10000, // Auto-refresh every 10 seconds
    }
  );
  const { data: swarmReviewJobsData, isLoading: swarmReviewJobsLoading, refetch: refetchSwarmReviewJobs } = useQuery(
    ['agent-jobs', 'swarm-review', swarmReviewVisibilityScope],
    () => apiClient.listAgentJobs({ page_size: 200, visibility_scope: swarmReviewVisibilityScope }),
    {
      enabled: activeTab === 'swarm',
      refetchInterval: 10000,
    }
  );

  const {
    data: deepLinkedJobData,
    error: deepLinkedJobError,
  } = useQuery(
    ['agent-job', deepLinkedJobId, 'deep-link'],
    () => apiClient.getAgentJob(String(deepLinkedJobId)),
    {
      enabled: !!deepLinkedJobId,
      retry: false,
      staleTime: 5000,
    }
  );

  const { data: checkpointQueueData, isLoading: checkpointQueueLoading, refetch: refetchCheckpointQueue } = useQuery(
    ['agent-checkpoint-queue', queueItemTypeFilter, queueStatusFilter, queueCustomerFilter, queueJobTypeFilter, queueSlaBucketFilter, queueEscalationFilter, queueOverdueOnly, queueSortBy],
    () => apiClient.getAgentCheckpointQueue({
      item_type: queueItemTypeFilter || undefined,
      status: queueStatusFilter || undefined,
      customer: queueCustomerFilter || undefined,
      job_type: queueJobTypeFilter || undefined,
      sla_bucket: queueSlaBucketFilter || undefined,
      escalation_level: queueEscalationFilter || undefined,
      overdue_only: queueOverdueOnly || undefined,
      sort_by: queueSortBy || undefined,
      limit: 100,
      offset: 0,
    }),
    {
      enabled: activeTab === 'queue',
      refetchInterval: 10000,
    }
  );

  const traceStartAt = useMemo(() => {
    if (traceDateRange === 'all') return undefined;
    const now = new Date();
    if (traceDateRange === '24h') now.setHours(now.getHours() - 24);
    else if (traceDateRange === '30d') now.setDate(now.getDate() - 30);
    else now.setDate(now.getDate() - 7);
    return now.toISOString();
  }, [traceDateRange]);

  const currentTraceViewFilters = useMemo(() => ({
    source_kind: traceSourceKindFilter || undefined,
    decision_type: traceDecisionTypeFilter || undefined,
    customer: traceCustomerFilter || undefined,
    status: traceStatusFilter || undefined,
    severity: traceSeverityFilter || undefined,
    actor_mode: traceActorModeFilter || undefined,
    triage_status: traceTriageStatusFilter || undefined,
    assigned_to_user_id: traceAssignedToUserIdFilter || undefined,
    unassigned_only: traceUnassignedOnly || undefined,
    escalation_state: traceEscalationStateFilter || undefined,
    pinned: tracePinnedOnly || undefined,
    actionable_only: traceActionableOnly || undefined,
    date_range: traceDateRange || undefined,
  }), [
    traceSourceKindFilter,
    traceDecisionTypeFilter,
    traceCustomerFilter,
    traceStatusFilter,
    traceSeverityFilter,
    traceActorModeFilter,
    traceTriageStatusFilter,
    traceAssignedToUserIdFilter,
    traceUnassignedOnly,
    traceEscalationStateFilter,
    tracePinnedOnly,
    traceActionableOnly,
    traceDateRange,
  ]);

  const buildTraceShareUrl = useCallback(
    (baseSearch?: string, traceEventId?: string) => {
      const params = new URLSearchParams(baseSearch ?? location.search);
      const setStringParam = (key: string, value: string) => {
        const text = String(value || '').trim();
        if (text) params.set(key, text);
        else params.delete(key);
      };
      const setBooleanParam = (key: string, value: boolean) => {
        if (value) params.set(key, 'true');
        else params.delete(key);
      };

      setStringParam('trace_source_kind', traceSourceKindFilter);
      setStringParam('trace_decision_type', traceDecisionTypeFilter);
      setStringParam('trace_customer', traceCustomerFilter);
      setStringParam('trace_status', traceStatusFilter);
      setStringParam('trace_severity', traceSeverityFilter);
      setStringParam('trace_actor_mode', traceActorModeFilter);
      setStringParam('trace_triage_status', traceTriageStatusFilter);
      setStringParam('trace_assigned_to_user_id', traceAssignedToUserIdFilter);
      setBooleanParam('trace_unassigned_only', traceUnassignedOnly);
      setStringParam('trace_escalation_state', traceEscalationStateFilter);
      setBooleanParam('trace_pinned', tracePinnedOnly);
      setBooleanParam('trace_actionable_only', traceActionableOnly);
      setStringParam('trace_date_range', traceDateRange || '7d');
      if (traceEventId !== undefined) {
        setStringParam('trace_event', traceEventId);
      }
      params.set('tab', 'trace');
      const qs = params.toString();
      return `${location.pathname}${qs ? `?${qs}` : ''}`;
    },
    [
      location.pathname,
      location.search,
      traceSourceKindFilter,
      traceDecisionTypeFilter,
      traceCustomerFilter,
      traceStatusFilter,
      traceSeverityFilter,
      traceActorModeFilter,
      traceTriageStatusFilter,
      traceAssignedToUserIdFilter,
      traceUnassignedOnly,
      traceEscalationStateFilter,
      tracePinnedOnly,
      traceActionableOnly,
      traceDateRange,
    ]
  );

  const { data: decisionTraceData, isLoading: decisionTraceLoading, refetch: refetchDecisionTrace } = useQuery(
    ['agent-decision-trace', traceSourceKindFilter, traceDecisionTypeFilter, traceCustomerFilter, traceStatusFilter, traceSeverityFilter, traceActorModeFilter, traceTriageStatusFilter, traceAssignedToUserIdFilter, traceUnassignedOnly, traceEscalationStateFilter, tracePinnedOnly, traceActionableOnly, traceDateRange, traceOffset],
    () => apiClient.getAgentDecisionTrace({
      source_kind: traceSourceKindFilter || undefined,
      decision_type: traceDecisionTypeFilter || undefined,
      customer: traceCustomerFilter || undefined,
      status: traceStatusFilter || undefined,
      severity: traceSeverityFilter || undefined,
      actor_mode: traceActorModeFilter || undefined,
      triage_status: traceTriageStatusFilter || undefined,
      assigned_to_user_id: traceAssignedToUserIdFilter || undefined,
      unassigned_only: traceUnassignedOnly || undefined,
      escalation_state: traceEscalationStateFilter || undefined,
      pinned: tracePinnedOnly || undefined,
      actionable_only: traceActionableOnly || undefined,
      start_at: traceStartAt,
      limit: 50,
      offset: traceOffset,
    }),
    {
      enabled: activeTab === 'trace',
      refetchInterval: 10000,
    }
  );

  const { data: decisionTraceAnalyticsData, isLoading: decisionTraceAnalyticsLoading, refetch: refetchDecisionTraceAnalytics } = useQuery<AgentDecisionTraceAnalyticsResponse>(
    [
      'agent-decision-trace-analytics',
      traceSourceKindFilter,
      traceDecisionTypeFilter,
      traceCustomerFilter,
      traceStatusFilter,
      traceSeverityFilter,
      traceActorModeFilter,
      traceTriageStatusFilter,
      traceAssignedToUserIdFilter,
      traceUnassignedOnly,
      traceEscalationStateFilter,
      tracePinnedOnly,
      traceActionableOnly,
      traceDateRange,
    ],
    () => apiClient.getAgentDecisionTraceAnalytics({
      source_kind: traceSourceKindFilter || undefined,
      decision_type: traceDecisionTypeFilter || undefined,
      customer: traceCustomerFilter || undefined,
      status: traceStatusFilter || undefined,
      severity: traceSeverityFilter || undefined,
      actor_mode: traceActorModeFilter || undefined,
      triage_status: traceTriageStatusFilter || undefined,
      assigned_to_user_id: traceAssignedToUserIdFilter || undefined,
      unassigned_only: traceUnassignedOnly || undefined,
      escalation_state: traceEscalationStateFilter || undefined,
      pinned: tracePinnedOnly || undefined,
      actionable_only: traceActionableOnly || undefined,
      start_at: traceStartAt,
      days: 7,
    }),
    {
      enabled: activeTab === 'trace',
      refetchInterval: 30000,
    }
  );

  const { data: traceViewsData } = useQuery(
    ['agent-decision-trace-views'],
    () => apiClient.listAgentDecisionTraceViews(),
    {
      enabled: activeTab === 'trace',
    }
  );

  const visibleDecisionTraceItems = useMemo(
    () => ((decisionTraceData?.items || []) as AgentDecisionTraceEvent[]).filter((event) => {
      if (traceOperatorPreset === 'compiler' && !isCompilerTraceEvent(event)) return false;
      if (traceOperatorPreset === 'approval_required') {
        const eventKind = String(event.event_type || event.decision_type || '').trim().toLowerCase();
        if (!isCompilerTraceEvent(event)) return false;
        if (!['follow_up_queued', 'follow_up_queued_for_approval'].includes(eventKind)) return false;
      }
      if (traceOperatorPreset === 'blocked_validation') {
        const eventKind = String(event.event_type || event.decision_type || '').trim().toLowerCase();
        if (!isCompilerTraceEvent(event)) return false;
        if (eventKind !== 'validation_blocked') return false;
      }
      if (traceOperatorPreset === 'failed_follow_up') {
        const eventKind = String(event.event_type || event.decision_type || '').trim().toLowerCase();
        if (!isCompilerTraceEvent(event)) return false;
        if (!['follow_up_failed', 'follow_up_cancelled'].includes(eventKind)) return false;
      }
      if (traceOperatorPreset === 'reevaluation_closeout') {
        const eventKind = String(event.event_type || event.decision_type || '').trim().toLowerCase();
        if (!eventKind.startsWith('reevaluation_')) return false;
      }
      return true;
    }),
    [decisionTraceData?.items, traceOperatorPreset]
  );

  const selectedTraceView = useMemo(
    () => (traceViewsData?.items || []).find((item) => item.id === selectedTraceViewId) || null,
    [selectedTraceViewId, traceViewsData?.items]
  );

  const markTraceFiltersDirty = useCallback(() => {
    traceFiltersDirtyRef.current = true;
  }, []);

  const applyTraceView = useCallback((view: AgentDecisionTraceView | null | undefined) => {
    const filters = (view?.filters || {}) as Record<string, any>;
    traceFiltersDirtyRef.current = false;
    setSelectedTraceViewId(String(view?.id || '').trim());
    setTraceViewNameDraft(String(view?.name || '').trim());
    setTraceViewIsDefaultDraft(Boolean(view?.is_default));
    setTraceSourceKindFilter(String(filters.source_kind || ''));
    setTraceDecisionTypeFilter(String(filters.decision_type || ''));
    setTraceCustomerFilter(String(filters.customer || ''));
    setTraceStatusFilter(String(filters.status || ''));
    setTraceSeverityFilter(String(filters.severity || ''));
    setTraceActorModeFilter(String(filters.actor_mode || ''));
    setTraceTriageStatusFilter(String(filters.triage_status || ''));
    setTraceAssignedToUserIdFilter(String(filters.assigned_to_user_id || ''));
    setTraceUnassignedOnly(Boolean(filters.unassigned_only));
    setTraceEscalationStateFilter(String(filters.escalation_state || ''));
    setTracePinnedOnly(Boolean(filters.pinned));
    setTraceActionableOnly(Boolean(filters.actionable_only));
    setTraceDateRange(String(filters.date_range || '7d'));
  }, []);

  const decisionTraceActionMutation = useMutation(
    ({ eventId, action, note, assigned_to_user_id, due_at }: { eventId: string; action: 'acknowledge' | 'start_investigation' | 'resolve' | 'reopen' | 'toggle_pin' | 'assign' | 'unassign' | 'set_due_at' | 'clear_due_at' | 'approve_launch' | 'reject_launch' | 'relaunch_follow_up'; note?: string; assigned_to_user_id?: string; due_at?: string }) =>
      apiClient.actionAgentDecisionTraceEvent(eventId, { action, note, assigned_to_user_id, due_at }),
    {
      onSuccess: (_res, vars) => {
        queryClient.invalidateQueries(['agent-decision-trace']);
        queryClient.invalidateQueries(['notifications']);
        queryClient.invalidateQueries(['notifications-unread-count']);
        if (vars?.action === 'approve_launch' || vars?.action === 'reject_launch' || vars?.action === 'relaunch_follow_up') {
          queryClient.invalidateQueries(['agent-checkpoint-queue']);
          queryClient.invalidateQueries(['research-portfolios']);
          queryClient.invalidateQueries(['domain-research-profiles']);
          queryClient.invalidateQueries(['research-inbox']);
          queryClient.invalidateQueries(['agent-jobs']);
          queryClient.invalidateQueries(['agent-jobs-stats']);
          toast.success(
            vars.action === 'approve_launch'
              ? 'Follow-up launched'
              : vars.action === 'reject_launch'
                ? 'Follow-up rejected'
                : 'Follow-up relaunched'
          );
        }
        if (vars?.action === 'resolve' || vars?.action === 'reopen') {
          setTraceActionNoteDrafts((current) => {
            const next = { ...current };
            delete next[String(vars.eventId || '')];
            return next;
          });
        }
        if (vars?.action === 'approve_launch' || vars?.action === 'reject_launch' || vars?.action === 'relaunch_follow_up') {
          setTraceActionNoteDrafts((current) => {
            const next = { ...current };
            delete next[String(vars.eventId || '')];
            return next;
          });
        }
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to update decision trace event');
      },
    }
  );

  const createTraceViewMutation = useMutation(
    (payload: { name: string; filters: Record<string, any>; is_default?: boolean }) => apiClient.createAgentDecisionTraceView(payload),
    {
      onSuccess: (view) => {
        queryClient.invalidateQueries(['agent-decision-trace-views']);
        setSelectedTraceViewId(String(view.id || ''));
        setTraceViewNameDraft(String(view.name || ''));
        setTraceViewIsDefaultDraft(Boolean(view.is_default));
        toast.success('Trace view saved');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to save trace view');
      },
    }
  );

  const updateTraceViewMutation = useMutation(
    ({ viewId, payload }: { viewId: string; payload: { name?: string; filters?: Record<string, any>; is_default?: boolean } }) =>
      apiClient.updateAgentDecisionTraceView(viewId, payload),
    {
      onSuccess: (view) => {
        queryClient.invalidateQueries(['agent-decision-trace-views']);
        setTraceViewNameDraft(String(view.name || ''));
        setTraceViewIsDefaultDraft(Boolean(view.is_default));
        toast.success('Trace view updated');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to update trace view');
      },
    }
  );

  const deleteTraceViewMutation = useMutation(
    (viewId: string) => apiClient.deleteAgentDecisionTraceView(viewId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agent-decision-trace-views']);
        setSelectedTraceViewId('');
        setTraceViewNameDraft('');
        setTraceViewIsDefaultDraft(false);
        toast.success('Trace view deleted');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to delete trace view');
      },
    }
  );

  const runDecisionTraceAction = useCallback((event: AgentDecisionTraceEvent, action: 'acknowledge' | 'start_investigation' | 'resolve' | 'reopen' | 'toggle_pin' | 'assign' | 'unassign' | 'set_due_at' | 'clear_due_at' | 'approve_launch' | 'reject_launch' | 'relaunch_follow_up', note?: string) => {
    if (event.is_derived) return;
    const normalizedNote = String(note || '').trim() || undefined;
    decisionTraceActionMutation.mutate({ eventId: event.event_id, action, note: normalizedNote });
  }, [decisionTraceActionMutation]);

  const downloadDecisionTraceExport = useCallback(
    async (format: 'json' | 'csv') => {
      try {
        await apiClient.downloadAgentDecisionTraceExport({
          format,
          source_kind: traceSourceKindFilter || undefined,
          decision_type: traceDecisionTypeFilter || undefined,
          customer: traceCustomerFilter || undefined,
          status: traceStatusFilter || undefined,
          severity: traceSeverityFilter || undefined,
          actor_mode: traceActorModeFilter || undefined,
          triage_status: traceTriageStatusFilter || undefined,
          assigned_to_user_id: traceAssignedToUserIdFilter || undefined,
          unassigned_only: traceUnassignedOnly || undefined,
          escalation_state: traceEscalationStateFilter || undefined,
          pinned: tracePinnedOnly || undefined,
          actionable_only: traceActionableOnly || undefined,
          start_at: traceStartAt,
        });
        toast.success(`Decision trace exported as ${format.toUpperCase()}`);
      } catch (error: any) {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to export decision trace');
      }
    },
    [
      traceSourceKindFilter,
      traceDecisionTypeFilter,
      traceCustomerFilter,
      traceStatusFilter,
      traceSeverityFilter,
      traceActorModeFilter,
      traceTriageStatusFilter,
      traceAssignedToUserIdFilter,
      traceUnassignedOnly,
      traceEscalationStateFilter,
      tracePinnedOnly,
      traceActionableOnly,
      traceStartAt,
    ]
  );

  const runDecisionTraceAssignmentAction = useCallback((event: AgentDecisionTraceEvent) => {
    if (event.is_derived) return;
    const assigned_to_user_id = String(traceAssigneeDrafts[event.event_id] ?? event.assigned_to_user_id ?? '').trim();
    if (!assigned_to_user_id) {
      toast.error('Select an assignee first');
      return;
    }
    decisionTraceActionMutation.mutate({ eventId: event.event_id, action: 'assign', assigned_to_user_id });
  }, [decisionTraceActionMutation, traceAssigneeDrafts]);

  const runDecisionTraceDueAtAction = useCallback((event: AgentDecisionTraceEvent) => {
    if (event.is_derived) return;
    const dueDraft = String(traceDueAtDrafts[event.event_id] ?? '').trim();
    if (!dueDraft) {
      toast.error('Enter a due date first');
      return;
    }
    const parsed = new Date(dueDraft);
    if (Number.isNaN(parsed.getTime())) {
      toast.error('Enter a valid due date');
      return;
    }
    decisionTraceActionMutation.mutate({ eventId: event.event_id, action: 'set_due_at', due_at: parsed.toISOString() });
  }, [decisionTraceActionMutation, traceDueAtDrafts]);

  // Deep-link: /autonomous-agents?job=<id>
  useEffect(() => {
    if (!landingTabInitializedRef.current && !deepLinkedTraceTab && !deepLinkedHealthTab && !deepLinkedQueueTab && !deepLinkedDomainTab && !deepLinkedFleetTab && !deepLinkedInboxTab && !deepLinkedJobId) {
      landingTabInitializedRef.current = true;
      setActiveTab('jobs');
    }
    if (deepLinkedTraceTab) {
      landingTabInitializedRef.current = true;
      setActiveTab('trace');
    }
    if (deepLinkedHealthTab) {
      landingTabInitializedRef.current = true;
      setActiveTab('health');
    }
    if (deepLinkedQueueTab) {
      landingTabInitializedRef.current = true;
      setActiveTab('queue');
    }
    if (deepLinkedDomainTab) {
      landingTabInitializedRef.current = true;
      setActiveTab('domain');
    }
    if (deepLinkedFleetTab) {
      landingTabInitializedRef.current = true;
      setActiveTab('fleet');
    }
    if (deepLinkedInboxTab) {
      landingTabInitializedRef.current = true;
      setActiveTab('inbox');
    }
    const normalizedInboxJobId = String(deepLinkedInboxJobId || '').trim();
    const normalizedInboxCustomer = String(deepLinkedInboxCustomer || '').trim();
    const normalizedQueueCustomer = String(deepLinkedQueueCustomer || '').trim();
    const normalizedQueueJobId = String(deepLinkedQueueJobId || '').trim();
    const normalizedHealthCustomer = String(deepLinkedHealthCustomer || '').trim();
    if (normalizedInboxJobId !== inboxJobFilter) {
      setInboxJobFilter(normalizedInboxJobId);
    }
    if (normalizedInboxCustomer !== inboxCustomerFilter) {
      setInboxCustomerFilter(normalizedInboxCustomer);
    }
    if (normalizedQueueCustomer !== queueCustomerFilter) {
      setQueueCustomerFilter(normalizedQueueCustomer);
    }
    if (normalizedQueueJobId !== queueJobFilter) {
      setQueueJobFilter(normalizedQueueJobId);
    }
    if (deepLinkedInboxHealthDrilldown !== inboxHealthDrilldown) {
      setInboxHealthDrilldown(deepLinkedInboxHealthDrilldown);
    }
    if (deepLinkedInboxPolicyDrilldown !== inboxPolicyDrilldown) {
      setInboxPolicyDrilldown(deepLinkedInboxPolicyDrilldown);
    }
    if (deepLinkedQueueHealthDrilldown !== queueHealthDrilldown) {
      setQueueHealthDrilldown(deepLinkedQueueHealthDrilldown);
    }
    if (normalizedHealthCustomer !== healthCustomerFilter) {
      setHealthCustomerFilter(normalizedHealthCustomer);
    }
    if (!deepLinkedJobId) return;
    if (deepLinkedJobData && String((deepLinkedJobData as any)?.id || '') === String(deepLinkedJobId)) {
      setSelectedJob(deepLinkedJobData as AgentJob);
      if (!deepLinkedQueueTab) setActiveTab('jobs');
      return;
    }

    const jobs = (jobsData as any)?.jobs || [];
    const match = jobs.find((j: any) => String(j.id) === String(deepLinkedJobId));
    if (match) {
      setSelectedJob(match);
      if (!deepLinkedQueueTab) setActiveTab('jobs');
      return;
    }

    const status = Number((deepLinkedJobError as any)?.response?.status || 0);
    if (status === 404) {
      // URL references a job that is no longer present (deleted/filtered out on server).
      // Clear stale selection token while preserving other query params.
      setSelectedJob(null);
      navigate(buildAutonomousAgentsUrl(), { replace: true });
    }
  }, [deepLinkedTraceTab, deepLinkedHealthTab, deepLinkedJobId, deepLinkedJobData, deepLinkedJobError, deepLinkedQueueTab, deepLinkedQueueCustomer, deepLinkedQueueJobId, deepLinkedQueueHealthDrilldown, deepLinkedDomainTab, deepLinkedFleetTab, deepLinkedInboxTab, deepLinkedInboxJobId, deepLinkedInboxCustomer, deepLinkedInboxHealthDrilldown, deepLinkedInboxPolicyDrilldown, deepLinkedHealthCustomer, healthCustomerFilter, inboxCustomerFilter, inboxHealthDrilldown, inboxPolicyDrilldown, inboxJobFilter, queueCustomerFilter, queueHealthDrilldown, queueJobFilter, jobsData, navigate, buildAutonomousAgentsUrl]);

  useEffect(() => {
    if (deepLinkedFleetId) {
      setExpandedPortfolioIds((prev) => (prev[deepLinkedFleetId] ? prev : { ...prev, [deepLinkedFleetId]: true }));
    }
  }, [deepLinkedFleetId]);

  useEffect(() => {
    if (deepLinkedProfileId) {
      setExpandedDomainProfileIds((prev) => (prev[deepLinkedProfileId] ? prev : { ...prev, [deepLinkedProfileId]: true }));
    }
  }, [deepLinkedProfileId]);

  useEffect(() => {
    if (deepLinkedJobId || selectedJob) return;
    const jobs = Array.isArray((jobsData as any)?.jobs) ? ((jobsData as any).jobs as AgentJob[]) : [];
    if (jobs.length > 0) {
      setSelectedJob(jobs[0]);
    }
  }, [deepLinkedJobId, jobsData, selectedJob]);

  useEffect(() => {
    setTraceOffset(0);
  }, [traceSourceKindFilter, traceDecisionTypeFilter, traceCustomerFilter, traceStatusFilter, traceSeverityFilter, traceActorModeFilter, traceTriageStatusFilter, traceAssignedToUserIdFilter, traceUnassignedOnly, traceEscalationStateFilter, tracePinnedOnly, traceActionableOnly, traceDateRange]);

  useEffect(() => {
    if (!deepLinkedTraceEventId) return;
    setActiveTab('trace');
    const matchingEvent = (decisionTraceData?.items || []).find((item) => item.event_id === deepLinkedTraceEventId);
    if (matchingEvent) {
      setExpandedTraceEventId(matchingEvent.event_id);
    }
  }, [deepLinkedTraceEventId, decisionTraceData?.items, setActiveTab]);

  useEffect(() => {
    if (selectedTraceViewId) return;
    if (traceFiltersDirtyRef.current) return;
    if (hasExplicitTraceContext) return;
    const defaultView = (traceViewsData?.items || []).find((item) => item.is_default);
    if (defaultView) {
      applyTraceView(defaultView);
    }
  }, [applyTraceView, hasExplicitTraceContext, selectedTraceViewId, traceViewsData?.items]);

  useLayoutEffect(() => {
    const params = new URLSearchParams(location.search);
    const hasTraceParams = TRACE_FILTER_QUERY_KEYS.some((key) => params.has(key));
    if (!hasTraceParams) return;

    const nextSourceKind = String(params.get('trace_source_kind') || '').trim();
    const nextDecisionType = String(params.get('trace_decision_type') || '').trim();
    const nextCustomer = String(params.get('trace_customer') || '').trim();
    const nextStatus = String(params.get('trace_status') || '').trim();
    const nextSeverity = String(params.get('trace_severity') || '').trim();
    const nextActorMode = String(params.get('trace_actor_mode') || '').trim();
    const nextTriageStatus = String(params.get('trace_triage_status') || '').trim();
    const nextAssignedToUserId = String(params.get('trace_assigned_to_user_id') || '').trim();
    const nextUnassignedOnly = ['1', 'true', 'yes'].includes(String(params.get('trace_unassigned_only') || '').trim().toLowerCase());
    const nextEscalationState = String(params.get('trace_escalation_state') || '').trim();
    const nextPinnedOnly = ['1', 'true', 'yes'].includes(String(params.get('trace_pinned') || '').trim().toLowerCase());
    const nextActionableOnly = ['1', 'true', 'yes'].includes(String(params.get('trace_actionable_only') || '').trim().toLowerCase());
    const nextDateRange = String(params.get('trace_date_range') || '7d').trim() || '7d';

    if (nextSourceKind !== traceSourceKindFilter) setTraceSourceKindFilter(nextSourceKind);
    if (nextDecisionType !== traceDecisionTypeFilter) setTraceDecisionTypeFilter(nextDecisionType);
    if (nextCustomer !== traceCustomerFilter) setTraceCustomerFilter(nextCustomer);
    if (nextStatus !== traceStatusFilter) setTraceStatusFilter(nextStatus);
    if (nextSeverity !== traceSeverityFilter) setTraceSeverityFilter(nextSeverity);
    if (nextActorMode !== traceActorModeFilter) setTraceActorModeFilter(nextActorMode);
    if (nextTriageStatus !== traceTriageStatusFilter) setTraceTriageStatusFilter(nextTriageStatus);
    if (nextAssignedToUserId !== traceAssignedToUserIdFilter) setTraceAssignedToUserIdFilter(nextAssignedToUserId);
    if (nextUnassignedOnly !== traceUnassignedOnly) setTraceUnassignedOnly(nextUnassignedOnly);
    if (nextEscalationState !== traceEscalationStateFilter) setTraceEscalationStateFilter(nextEscalationState);
    if (nextPinnedOnly !== tracePinnedOnly) setTracePinnedOnly(nextPinnedOnly);
    if (nextActionableOnly !== traceActionableOnly) setTraceActionableOnly(nextActionableOnly);
    if (nextDateRange !== traceDateRange) setTraceDateRange(nextDateRange);
  }, [
    activeTab,
    location.search,
    traceSourceKindFilter,
    traceDecisionTypeFilter,
    traceCustomerFilter,
    traceStatusFilter,
    traceSeverityFilter,
    traceActorModeFilter,
    traceTriageStatusFilter,
    traceAssignedToUserIdFilter,
    traceUnassignedOnly,
    traceEscalationStateFilter,
    tracePinnedOnly,
    traceActionableOnly,
    traceDateRange,
  ]);

  useEffect(() => {
    if (activeTab !== 'trace') return;
    if (!selectedTraceViewId && !hasExplicitTraceFilterParams && !traceFiltersDirtyRef.current) return;
    const nextSearch = buildTraceShareUrl(location.search);
    const nextSearchOnly = nextSearch.includes('?') ? nextSearch.slice(nextSearch.indexOf('?')) : '';
    if (canonicalizeSearchParams(location.search) === canonicalizeSearchParams(nextSearchOnly)) return;
    navigate(nextSearch, { replace: true });
  }, [activeTab, buildTraceShareUrl, hasExplicitTraceFilterParams, location.pathname, location.search, navigate, selectedTraceViewId]);

  useEffect(() => {
    if (activeTab !== 'trace') return;
    if (!selectedTraceViewId || !selectedTraceView) return;
    if (traceViewFiltersMatch(currentTraceViewFilters, selectedTraceView.filters as Record<string, any>)) return;
    setSelectedTraceViewId('');
    setTraceViewNameDraft('');
    setTraceViewIsDefaultDraft(false);
  }, [activeTab, currentTraceViewFilters, selectedTraceView, selectedTraceViewId]);

  // Deep-link controls:
  // - graph: ?gh=critical|warning|ok|unknown&gsort=graph_health_critical_first|graph_severity_desc|scope_guard_blocked_first|experiment_recovery_priority
  // - launch mode: ?lm=quick_start_claude_backend|quick_start_role_workflow
  // - relaunch children: ?rhc=yes|no
  // - relaunch parent: ?rfj=<uuid>
  // - memory dedup skipped filter: ?mdf=gt0|gte3|gte5
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const gh = String(params.get('gh') || '').toLowerCase();
    const gsort = String(params.get('gsort') || '').toLowerCase();
    const lmRaw = String(params.get('lm') || '').trim().toLowerCase();
    const rhcRaw = String(params.get('rhc') || '').trim().toLowerCase();
    const rfjRaw = String(params.get('rfj') || '').trim();
    const mdfRaw = String(params.get('mdf') || '').trim().toLowerCase();
    const queueItemTypeRaw = String(params.get('queue_item_type') || '').trim().toLowerCase();
    const queueCustomerRaw = String(params.get('queue_customer') || '').trim();
    const queueJobRaw = String(params.get('queue_job') || '').trim();
    const queueHealthDrilldownRaw = String(params.get('queue_health_drilldown') || '').trim().toLowerCase();
    const queueSlaRaw = String(params.get('queue_sla') || '').trim().toLowerCase();
    const allowedHealth = new Set(['', 'critical', 'warning', 'ok', 'unknown']);
    const allowedSort = new Set(['none', 'graph_health_critical_first', 'graph_severity_desc', 'scope_guard_blocked_first', 'experiment_recovery_priority']);
    const allowedDedup = new Set(['', 'gt0', 'gte3', 'gte5']);
    const allowedQueueType = new Set(['', 'approval_checkpoint', 'job_recovery', 'follow_up_recommendation', 'policy_review', 'budget_review']);
    const allowedQueueSla = new Set(['', 'normal', 'at_risk', 'overdue']);
    const normalizedRhc = rhcRaw === 'yes' || rhcRaw === 'true'
      ? 'yes'
      : rhcRaw === 'no' || rhcRaw === 'false'
        ? 'no'
        : '';
    const normalizedRfj = /^[0-9a-fA-F-]{36}$/.test(rfjRaw) ? rfjRaw : '';
    const normalizedLm = (() => {
      if (!lmRaw) return '';
      if (lmRaw === '__none__' || lmRaw === 'none' || lmRaw === 'manual') return '__none__';
      return /^[a-z0-9_:-]{2,80}$/.test(lmRaw) ? lmRaw : '';
    })();
    const nextHealth = allowedHealth.has(gh) ? gh : '';
    const nextSort = allowedSort.has(gsort) ? gsort : 'none';
    const nextDedup = allowedDedup.has(mdfRaw) ? mdfRaw : '';
    const nextQueueType = allowedQueueType.has(queueItemTypeRaw) ? queueItemTypeRaw : '';
    const nextQueueSla = allowedQueueSla.has(queueSlaRaw) ? queueSlaRaw : '';
    const nextQueueHealthDrilldown = normalizeQueueHealthDrilldown(queueHealthDrilldownRaw);
    setLaunchModeFilter((current) => (normalizedLm === current ? current : normalizedLm));
    setHasRelaunchChildrenFilter((current) => (normalizedRhc === current ? current : normalizedRhc));
    setRelaunchFromJobIdFilter((current) => (normalizedRfj === current ? current : normalizedRfj));
    setGraphHealthFilter((current) => (nextHealth === current ? current : nextHealth));
    setGraphSortBy((current) => (nextSort === current ? current : nextSort));
    setDedupSkipFilter((current) => (nextDedup === current ? current : nextDedup));
    setQueueItemTypeFilter((current) => (nextQueueType === current ? current : nextQueueType));
    setQueueCustomerFilter((current) => (queueCustomerRaw === current ? current : queueCustomerRaw));
    setQueueJobFilter((current) => (queueJobRaw === current ? current : queueJobRaw));
    setQueueHealthDrilldown((current) => (
      nextQueueHealthDrilldown === current ? current : nextQueueHealthDrilldown
    ));
    setQueueSlaBucketFilter((current) => (nextQueueSla === current ? current : nextQueueSla));
  }, [location.search]);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const currentHealth = String(params.get('gh') || '').toLowerCase();
    const currentSort = String(params.get('gsort') || '').toLowerCase() || 'none';
    const currentLmRaw = String(params.get('lm') || '').trim().toLowerCase();
    const currentRhcRaw = String(params.get('rhc') || '').trim().toLowerCase();
    const currentRfjRaw = String(params.get('rfj') || '').trim();
    const currentRhc = currentRhcRaw === 'yes' || currentRhcRaw === 'true'
      ? 'yes'
      : currentRhcRaw === 'no' || currentRhcRaw === 'false'
        ? 'no'
        : '';
    const currentRfj = /^[0-9a-fA-F-]{36}$/.test(currentRfjRaw) ? currentRfjRaw : '';
    const currentLm = (currentLmRaw === '__none__' || currentLmRaw === 'none' || currentLmRaw === 'manual')
      ? '__none__'
      : currentLmRaw;
    const targetHealth = String(graphHealthFilter || '').toLowerCase();
    const targetSort = String(graphSortBy || 'none').toLowerCase();
    const targetLmState = String(launchModeFilter || '').trim().toLowerCase();
    const targetRhc = String(hasRelaunchChildrenFilter || '').trim().toLowerCase();
    const targetRfjRaw = String(relaunchFromJobIdFilter || '').trim();
    const targetDedup = String(dedupSkipFilter || '').trim().toLowerCase();
    const targetQueueCustomer = String(queueCustomerFilter || '').trim();
    const targetQueueJob = String(queueJobFilter || '').trim();
    const targetQueueHealthDrilldown = String(queueHealthDrilldown || '').trim().toLowerCase();
    const targetRfj = /^[0-9a-fA-F-]{36}$/.test(targetRfjRaw) ? targetRfjRaw : '';
    const targetLm = targetLmState === '__none__' ? 'none' : targetLmState;
    const targetLmCompare = targetLmState === '__none__' ? '__none__' : targetLmState;
    if (
      deepLinkedQueueTab
      && (
        String(deepLinkedQueueCustomer || '').trim() !== targetQueueCustomer
        || String(deepLinkedQueueJobId || '').trim() !== targetQueueJob
        || deepLinkedQueueHealthDrilldown !== targetQueueHealthDrilldown
      )
    ) {
      return;
    }
    if (
      currentHealth === targetHealth &&
      currentSort === targetSort &&
      currentLm === targetLmCompare &&
      currentRhc === targetRhc &&
      currentRfj === targetRfj &&
      String(params.get('mdf') || '').trim().toLowerCase() === targetDedup &&
      String(params.get('queue_customer') || '').trim() === targetQueueCustomer &&
      String(params.get('queue_job') || '').trim() === targetQueueJob &&
      String(params.get('queue_health_drilldown') || '').trim().toLowerCase() === targetQueueHealthDrilldown
    ) return;

    if (targetHealth) params.set('gh', targetHealth);
    else params.delete('gh');
    if (targetSort && targetSort !== 'none') params.set('gsort', targetSort);
    else params.delete('gsort');
    if (targetLm) params.set('lm', targetLm);
    else params.delete('lm');
    if (targetRhc) params.set('rhc', targetRhc);
    else params.delete('rhc');
    if (targetRfj) params.set('rfj', targetRfj);
    else params.delete('rfj');
    if (targetDedup) params.set('mdf', targetDedup);
    else params.delete('mdf');
    if (targetQueueCustomer) params.set('queue_customer', targetQueueCustomer);
    else params.delete('queue_customer');
    if (targetQueueJob) params.set('queue_job', targetQueueJob);
    else params.delete('queue_job');
    if (targetQueueHealthDrilldown) params.set('queue_health_drilldown', targetQueueHealthDrilldown);
    else params.delete('queue_health_drilldown');

    const search = params.toString();
    navigate(`${location.pathname}${search ? `?${search}` : ''}`, { replace: true });
  }, [deepLinkedQueueCustomer, deepLinkedQueueHealthDrilldown, deepLinkedQueueJobId, deepLinkedQueueTab, launchModeFilter, hasRelaunchChildrenFilter, relaunchFromJobIdFilter, dedupSkipFilter, graphHealthFilter, graphSortBy, location.pathname, location.search, navigate, queueCustomerFilter, queueHealthDrilldown, queueJobFilter]);

  useEffect(() => {
    const jobId = String(selectedJob?.id || '').trim();
    const status = String(selectedJob?.status || '').toLowerCase();
    if (!jobId || TERMINAL_JOB_STATUSES.has(status)) return;

    let closed = false;
    let ws: WebSocket | null = null;

    try {
      ws = apiClient.createAgentJobProgressWebSocket(jobId);
    } catch (error) {
      console.error('Failed to create agent job progress websocket:', error);
      return;
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as AgentJobProgressUpdate | Record<string, any>;
        if (!data || typeof data !== 'object') return;
        if (String((data as any).type || '') !== 'progress') return;

        setSelectedJob((prev) => {
          if (!prev || String(prev.id) !== jobId) return prev;
          return mergeProgressUpdateIntoJob(prev, data as AgentJobProgressUpdate);
        });

        queryClient.setQueriesData(['agent-jobs'], (prev: any) => {
          if (!prev || !Array.isArray(prev.jobs)) return prev;
          return {
            ...prev,
            jobs: prev.jobs.map((row: AgentJob) =>
              String(row?.id || '') === jobId
                ? mergeProgressUpdateIntoJob(row, data as AgentJobProgressUpdate)
                : row
            ),
          };
        });
        queryClient.setQueryData(['agent-job', jobId, 'deep-link'], (prev: any) => {
          if (!prev || String(prev.id || '') !== jobId) return prev;
          return mergeProgressUpdateIntoJob(prev as AgentJob, data as AgentJobProgressUpdate);
        });

        const nextStatus = String((data as any).status || '').toLowerCase();
        if (TERMINAL_JOB_STATUSES.has(nextStatus)) {
          queryClient.invalidateQueries(['agent-jobs']);
          queryClient.invalidateQueries(['agent-jobs-stats']);
          if (!closed) {
            closed = true;
            ws?.close();
          }
        }
      } catch (error) {
        console.error('Failed to parse job progress websocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('Agent job progress websocket error:', error);
    };

    return () => {
      if (!closed) {
        closed = true;
        ws?.close();
      }
    };
  }, [selectedJob?.id, selectedJob?.status, queryClient]);

  // Fetch stats
  const { data: stats } = useQuery(
    ['agent-jobs-stats'],
    () => apiClient.getAgentJobStats(),
    {
      refetchInterval: 30000,
    }
  );

  // Fetch templates
  const { data: templatesData } = useQuery(
    ['agent-job-templates', templateRecommendScope, templateRecommendGoal],
    () =>
      apiClient.listAgentJobTemplates(undefined, {
        recommend_scope: templateRecommendScope || undefined,
        recommend_goal: templateRecommendGoal.trim() || undefined,
      }),
    {
      enabled: activeTab === 'templates',
    }
  );
  const { data: documentSources } = useQuery(
    ['document-sources', 'all'],
    () => apiClient.getDocumentSources(),
    { staleTime: 30000 }
  );
  const codeSources = useMemo(() => {
    const items = (documentSources || []) as any[];
    return items.filter((s) => ['github', 'gitlab'].includes(String(s?.source_type || s?.sourceType || '').toLowerCase()));
  }, [documentSources]);
  const { data: codingSwarmProfilesData } = useQuery(
    ['coding-swarm-profiles'],
    () =>
      typeof (apiClient as any).listCodingSwarmProfiles === 'function'
        ? apiClient.listCodingSwarmProfiles({ limit: 200, offset: 0 })
        : Promise.resolve({ items: [], total: 0, limit: 200, offset: 0 }),
    { staleTime: 30000 }
  );
  const codingSwarmProfiles = useMemo(
    () => (((codingSwarmProfilesData as any)?.items || []) as CodingSwarmProfile[]),
    [codingSwarmProfilesData]
  );
  const { data: collaborationUsersData } = useQuery(
    ['collaboration-users'],
    () =>
      typeof (apiClient as any).listCollaborationUsers === 'function'
        ? apiClient.listCollaborationUsers('', 1, 100)
        : Promise.resolve({ items: [], total: 0, page: 1, page_size: 100 }),
    { staleTime: 30000 }
  );
  const collaborationUsers = useMemo(
    () => (((collaborationUsersData as any)?.items || []) as User[]),
    [collaborationUsersData]
  );
  const collaborationUserById = useMemo(
    () => Object.fromEntries(collaborationUsers.map((candidate) => [String(candidate.id), candidate] as const)) as Record<string, User>,
    [collaborationUsers]
  );
  const userLabelById = useCallback(
    (candidateId?: string | null) => {
      const normalized = String(candidateId || '').trim();
      if (!normalized) return '';
      if (normalized === String(user?.id || '')) return 'You';
      const candidate = collaborationUserById[normalized];
      return String(candidate?.full_name || candidate?.username || candidate?.email || normalized).trim();
    },
    [collaborationUserById, user]
  );
  const codeSourceById = useMemo(
    () =>
      Object.fromEntries(
        codeSources.map((source: any) => [String(source.id), source] as const)
      ) as Record<string, any>,
    [codeSources]
  );
  const filteredCodingSwarmProfiles = useMemo(
    () =>
      codingSwarmProfiles.filter((profile) => {
        if (profilePresetFilter && String(profile.preset_key || '') !== profilePresetFilter) return false;
        if (profileSourceFilter && String(profile.source_id || '') !== profileSourceFilter) return false;
        if (profileStatusFilter && String(profile.status || '').toLowerCase() !== profileStatusFilter) return false;
        if (profileDefaultOnly && !profile.is_default) return false;
        if (profileVisibilityFilter && String(profile.visibility || 'private').toLowerCase() !== profileVisibilityFilter) return false;
        if (profileOwnershipFilter === 'mine' && String(profile.user_id || '') !== String(user?.id || '')) return false;
        if (profileOwnershipFilter === 'shared' && String(profile.user_id || '') === String(user?.id || '')) return false;
        if (profileOwnerFilter && String(profile.user_id || '') !== profileOwnerFilter) return false;
        return true;
      }),
    [codingSwarmProfiles, profilePresetFilter, profileSourceFilter, profileStatusFilter, profileDefaultOnly, profileVisibilityFilter, profileOwnershipFilter, profileOwnerFilter, user]
  );
  useEffect(() => {
    if (!backlogSourceId && codeSources.length > 0) {
      setBacklogSourceId(String((codeSources[0] as any)?.id || ''));
    }
  }, [backlogSourceId, codeSources]);
  const claudeBackendTemplate = useMemo(
    () =>
      (((templatesData as any)?.templates || []) as AgentJobTemplate[]).find(
        (t) => String(t?.name || '').toLowerCase() === 'claude_code_backend'
      ) || null,
    [templatesData]
  );

  // Fetch chain definitions
  const { data: chainsData } = useQuery(
    ['agent-job-chains'],
    () => apiClient.listChainDefinitions()
  );
  const displayedChainDefinitions = useMemo(() => {
    const chains = (((chainsData as any)?.chains || []) as AgentJobChainDefinition[]).slice();
    const isRecoveryPlaybook = (chain: AgentJobChainDefinition) => {
      const name = String(chain.name || '').toLowerCase();
      const displayName = String(chain.display_name || '').toLowerCase();
      const description = String(chain.description || '').toLowerCase();
      return (
        name.startsWith('playbook_recovery_')
        || displayName.includes('recovery playbook')
        || description.includes('recovery playbook')
        || description.includes('saved as a recovery playbook')
      );
    };
    return chains.sort((a, b) => {
      const aRecovery = isRecoveryPlaybook(a) ? 0 : 1;
      const bRecovery = isRecoveryPlaybook(b) ? 0 : 1;
      if (aRecovery !== bRecovery) return aRecovery - bRecovery;
      const aSystem = a.is_system ? 0 : 1;
      const bSystem = b.is_system ? 0 : 1;
      if (aSystem !== bSystem) return aSystem - bSystem;
      return String(a.display_name || a.name || '').localeCompare(String(b.display_name || b.name || ''));
    });
  }, [chainsData]);
  const { data: codingBacklogData, isLoading: codingBacklogLoading, refetch: refetchCodingBacklog } = useQuery(
    ['coding-backlog-items', activeTab === 'backlog' ? backlogVisibilityScope : 'all', activeTab === 'backlog' ? backlogAssignmentFilter : ''],
    () => apiClient.listCodingBacklogItems({
      limit: 100,
      offset: 0,
      visibility_scope: activeTab === 'backlog' ? backlogVisibilityScope : 'all',
      assigned_user_id: activeTab === 'backlog' && backlogAssignmentFilter ? backlogAssignmentFilter : undefined,
    }),
    {
      enabled: activeTab === 'backlog' || activeTab === 'swarm' || activeTab === 'outcomes',
      refetchInterval: 15000,
    }
  );
  const { data: swarmAnalyticsData, isLoading: swarmAnalyticsLoading, refetch: refetchSwarmAnalytics } = useQuery(
    ['agent-job-swarm-analytics', swarmReviewVisibilityScope],
    () => apiClient.getAgentJobSwarmAnalytics({ visibility_scope: swarmReviewVisibilityScope }),
    {
      enabled: activeTab === 'swarm',
      refetchInterval: 15000,
    }
  );
  const swarmOutcomeDateFrom = useMemo(() => {
    if (swarmOutcomeDateRange === '7d') {
      return new Date(Date.now() - (7 * 24 * 60 * 60 * 1000)).toISOString();
    }
    if (swarmOutcomeDateRange === '30d') {
      return new Date(Date.now() - (30 * 24 * 60 * 60 * 1000)).toISOString();
    }
    return undefined;
  }, [swarmOutcomeDateRange]);
  const { data: swarmOutcomeAnalyticsData, isLoading: swarmOutcomeAnalyticsLoading, refetch: refetchSwarmOutcomeAnalytics } = useQuery(
    ['agent-job-swarm-outcomes', swarmOutcomePresetFilter, swarmOutcomeTerminalFilter, swarmOutcomePromotionFilter, swarmOutcomeDateFrom, swarmOutcomeVisibilityScope],
    () =>
      apiClient.getAgentJobSwarmOutcomeAnalytics({
        preset_key: swarmOutcomePresetFilter || undefined,
        terminal_outcome: swarmOutcomeTerminalFilter || undefined,
        promotion_mode: swarmOutcomePromotionFilter || undefined,
        visibility_scope: swarmOutcomeVisibilityScope,
        date_from: swarmOutcomeDateFrom,
      }),
    {
      enabled: ['outcomes', 'jobs', 'swarm', 'backlog'].includes(activeTab),
      refetchInterval: activeTab === 'outcomes' ? 15000 : false,
    }
  );
  const { data: domainProfilesData, isLoading: domainProfilesLoading, refetch: refetchDomainProfiles } = useQuery(
    ['domain-research-profiles'],
    () => apiClient.listDomainResearchProfiles({ limit: 100, offset: 0 }),
    {
      enabled: activeTab === 'domain' || activeTab === 'fleet',
      refetchInterval: 15000,
    }
  );
  const { data: scientificSandboxProfilesData } = useQuery(
    ['scientific-sandbox-profiles', isAdmin ? 'include-disabled' : 'enabled-only'],
    () => apiClient.listScientificSandboxProfiles(isAdmin ? { include_disabled: true } : undefined),
    {
      staleTime: 60000,
    }
  );
  const { data: researchPortfoliosData, isLoading: researchPortfoliosLoading, refetch: refetchResearchPortfolios } = useQuery(
    ['research-portfolios'],
    () => apiClient.listResearchPortfolios({ limit: 100, offset: 0 }),
    {
      enabled: activeTab === 'fleet',
      refetchInterval: 15000,
    }
  );
  const domainProfileById = useMemo(
    () =>
      Object.fromEntries(
        ((((domainProfilesData as any)?.items || []) as DomainResearchProfile[]).map((profile) => [String(profile.id), profile]))
      ) as Record<string, DomainResearchProfile>,
    [domainProfilesData]
  );
  const createCompilerArtifactMutation = useMutation(
    (data: {
      job_type: 'compiler_regression_explanation' | 'compiler_patch_proposal' | 'compiler_patch_draft';
      title: string;
      document_ids: string[];
      research_note_id?: string;
      experiment_run_ids?: string[];
      primary_run_id?: string;
      comparison_run_id?: string;
      source_id?: string;
      output_format?: string;
      output_style?: string;
    }) => apiClient.createSynthesisJob(data),
    {
      onSuccess: (job: any) => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['research-portfolios']);
        toast.success(`Started ${String(job?.job_type || 'compiler artifact').replace(/_/g, ' ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to create compiler artifact');
      },
    }
  );
  const saveCompilerArtifactNoteMutation = useMutation(
    ({ jobId }: { jobId: string }) => apiClient.saveSynthesisJobAsResearchNote(jobId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['research-portfolios']);
        toast.success('Compiler artifact note saved');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to save compiler artifact note');
      },
    }
  );

  useEffect(() => {
    const isFleetTarget = activeTab === 'fleet' && !!deepLinkedFleetId;
    const isDomainTarget = activeTab === 'domain' && !!deepLinkedProfileId;
    if (!isFleetTarget && !isDomainTarget) return;
    const scope = isFleetTarget ? 'fleet' : 'domain';
    const ownerId = isFleetTarget ? deepLinkedFleetId : deepLinkedProfileId;
    if (!ownerId) return;
    const cardKey = buildAutonomyCardKey(scope, ownerId);
    const cardNode = autonomyTargetCardRefs.current[cardKey];
    const opportunityId = deepLinkedOpportunityId;
    const candidateRowKeys = opportunityId
      ? [
          buildAutonomyOpportunityRowKey(scope, ownerId, opportunityId),
          buildAutonomyReviewRowKey(scope, ownerId, 'pending', opportunityId),
          buildAutonomyReviewRowKey(scope, ownerId, 'manual', opportunityId),
          buildAutonomyReviewRowKey(scope, ownerId, 'suppressed', opportunityId),
        ]
      : [];
    const rowKey = candidateRowKeys.find((key) => autonomyTargetRowRefs.current[key]);
    const rowNode = rowKey ? autonomyTargetRowRefs.current[rowKey] : null;
    const targetNode = rowNode || cardNode;
    if (!targetNode) return;
    if (scope === 'fleet') {
      setExpandedPortfolioIds((prev) => (prev[ownerId] ? prev : { ...prev, [ownerId]: true }));
    } else {
      setExpandedDomainProfileIds((prev) => (prev[ownerId] ? prev : { ...prev, [ownerId]: true }));
    }
    targetNode.scrollIntoView?.({ block: 'center', behavior: 'auto' });
    if (rowKey) {
      setHighlightedAutonomyRowKey(rowKey);
      setHighlightedAutonomyCardKey('');
    } else {
      setHighlightedAutonomyRowKey('');
      setHighlightedAutonomyCardKey(cardKey);
    }
    const timer = window.setTimeout(() => {
      setHighlightedAutonomyRowKey((current) => (rowKey && current === rowKey ? '' : current));
      setHighlightedAutonomyCardKey((current) => (!rowKey && current === cardKey ? '' : current));
    }, 2200);
    return () => window.clearTimeout(timer);
  }, [
    activeTab,
    buildAutonomyCardKey,
    buildAutonomyOpportunityRowKey,
    buildAutonomyReviewRowKey,
    deepLinkedFleetId,
    deepLinkedOpportunityId,
    deepLinkedProfileId,
    domainProfilesData,
    researchPortfoliosData,
  ]);

  const scientificSandboxProfiles = useMemo(
    () => (((scientificSandboxProfilesData as any)?.items || []) as ScientificSandboxProfile[]),
    [scientificSandboxProfilesData]
  );
  const scientificSandboxProfileById = useMemo(
    () =>
      Object.fromEntries(
        scientificSandboxProfiles.map((profile) => [String(profile.id), profile] as const)
      ) as Record<string, ScientificSandboxProfile>,
    [scientificSandboxProfiles]
  );
  const resolveSandboxProfileId = useCallback(
    (trackType: string) => {
      const normalizedTrackType = String(trackType || 'generic').trim().toLowerCase() || 'generic';
      const exactDefault = scientificSandboxProfiles.find(
        (profile) => String(profile.track_type || '').trim().toLowerCase() === normalizedTrackType && profile.is_default
      );
      if (exactDefault?.id) return String(exactDefault.id);
      const exactEnabled = scientificSandboxProfiles.find(
        (profile) => String(profile.track_type || '').trim().toLowerCase() === normalizedTrackType && profile.enabled
      );
      if (exactEnabled?.id) return String(exactEnabled.id);
      const genericDefault = scientificSandboxProfiles.find(
        (profile) => String(profile.track_type || '').trim().toLowerCase() === 'generic' && profile.is_default
      );
      if (genericDefault?.id) return String(genericDefault.id);
      if (normalizedTrackType === 'compiler') return 'scientific-compiler-sandbox';
      if (normalizedTrackType === 'microarchitecture') return 'scientific-microarchitecture-sandbox';
      return 'scientific-generic-sandbox';
    },
    [scientificSandboxProfiles]
  );
  const visibleScientificSandboxProfiles = useMemo(
    () =>
      scientificSandboxProfiles.filter(
        (profile) => isAdmin || profile.enabled
      ),
    [isAdmin, scientificSandboxProfiles]
  );
  const domainAvailableSandboxProfiles = useMemo(
    () =>
      visibleScientificSandboxProfiles.filter((profile) => {
        const track = String(profile.track_type || '').trim().toLowerCase();
        return profile.enabled && (track === String(domainProfileTrackType).toLowerCase() || track === 'generic');
      }),
    [domainProfileTrackType, visibleScientificSandboxProfiles]
  );
  const portfolioAvailableSandboxProfiles = useMemo(
    () => visibleScientificSandboxProfiles.filter((profile) => profile.enabled),
    [visibleScientificSandboxProfiles]
  );
  useEffect(() => {
    if (!domainAvailableSandboxProfiles.length) {
      setDomainProfileSandboxProfileId(resolveSandboxProfileId(domainProfileTrackType));
      return;
    }
    if (
      !domainProfileSandboxProfileId ||
      !domainAvailableSandboxProfiles.some((profile) => String(profile.id) === String(domainProfileSandboxProfileId))
    ) {
      setDomainProfileSandboxProfileId(resolveSandboxProfileId(domainProfileTrackType));
    }
  }, [domainAvailableSandboxProfiles, domainProfileSandboxProfileId, domainProfileTrackType, resolveSandboxProfileId]);
  useEffect(() => {
    if (!portfolioAvailableSandboxProfiles.length) {
      setPortfolioSandboxProfileId(resolveSandboxProfileId('compiler'));
      return;
    }
    if (
      !portfolioSandboxProfileId ||
      !portfolioAvailableSandboxProfiles.some((profile) => String(profile.id) === String(portfolioSandboxProfileId))
    ) {
      setPortfolioSandboxProfileId(resolveSandboxProfileId('compiler'));
    }
  }, [portfolioAvailableSandboxProfiles, portfolioSandboxProfileId, resolveSandboxProfileId]);

  // Research Inbox
  const { data: inboxStats } = useQuery(
    ['research-inbox-stats'],
    () => apiClient.getResearchInboxStats(),
    {
      refetchInterval: 20000,
    }
  );

  const { data: inboxData, isLoading: inboxLoading, refetch: refetchInbox } = useQuery(
    ['research-inbox', inboxStatusFilter, inboxTypeFilter, inboxCustomerFilter, inboxSearch, inboxJobFilter],
    () =>
      apiClient.listResearchInboxItems({
        status: inboxStatusFilter || undefined,
        item_type: inboxTypeFilter || undefined,
        customer: inboxCustomerFilter || undefined,
        job_id: inboxJobFilter || undefined,
        q: inboxSearch.trim() || undefined,
        limit: 100,
        offset: 0,
      }),
    {
      enabled: activeTab === 'inbox',
      refetchInterval: 15000,
    }
  );

  const { data: monitorProfiles, isLoading: monitorProfilesLoading, refetch: refetchMonitorProfiles } = useQuery(
    ['research-monitor-profiles'],
    () => apiClient.listResearchMonitorProfiles(),
    {
      enabled: showMonitorProfilesModal,
      staleTime: 30000,
    }
  );

  const { data: monitorAnalyticsData, isLoading: monitorAnalyticsLoading, refetch: refetchMonitorAnalytics } = useQuery(
    ['research-monitor-analytics'],
    () => apiClient.getResearchMonitorAnalytics(),
    {
      enabled: activeTab === 'health',
      staleTime: 30000,
    }
  );

  const healthCustomers = useMemo(
    () =>
      Array.from(
        new Set(
          ((monitorAnalyticsData as ResearchMonitorAnalyticsResponse | undefined)?.monitors || [])
            .map((monitor) => String(monitor.customer || '').trim())
            .filter(Boolean)
        )
      ).sort((a, b) => a.localeCompare(b)),
    [monitorAnalyticsData]
  );

  const customerPortfolioRows = useMemo(
    () => ((monitorAnalyticsData as ResearchMonitorAnalyticsResponse | undefined)?.customers || []),
    [monitorAnalyticsData]
  );
  const selectedPortfolioProfileIds = useMemo(
    () => Object.entries(portfolioProfileSelection).filter(([, enabled]) => enabled).map(([id]) => id),
    [portfolioProfileSelection]
  );
  const selectedDomainProfileRepoSourceIds = useMemo(
    () => Object.entries(domainProfileRepoSelection).filter(([, enabled]) => enabled).map(([id]) => id),
    [domainProfileRepoSelection]
  );
  const filteredScientificSandboxProfiles = useMemo(
    () =>
      visibleScientificSandboxProfiles.filter((profile) =>
        isAdmin ? (showDisabledSandboxProfiles ? true : profile.enabled) : profile.enabled
      ),
    [isAdmin, showDisabledSandboxProfiles, visibleScientificSandboxProfiles]
  );
  const editingScientificSandboxProfile = useMemo(
    () =>
      editingScientificSandboxProfileId
        ? scientificSandboxProfileById[String(editingScientificSandboxProfileId)] || null
        : null,
    [editingScientificSandboxProfileId, scientificSandboxProfileById]
  );
  const editingScientificSandboxSystemManaged = Boolean(editingScientificSandboxProfile?.system_managed);
  const resetScientificSandboxDraft = useCallback(() => {
    setEditingScientificSandboxProfileId('');
    setSandboxProfileDraft(buildScientificSandboxProfileDraft());
  }, []);
  const openResearchNote = useCallback((noteId?: string | null) => {
    const normalized = String(noteId || '').trim();
    if (!normalized) return;
    navigate(`/research-notes?note=${encodeURIComponent(normalized)}`);
  }, [navigate]);
  const runCompilerArtifactAction = useCallback((
    run: ScientificValidationRunSummary,
    action: 'create_regression_explanation' | 'create_patch_proposal' | 'create_patch_draft',
    ownerProfile?: DomainResearchProfile | null,
  ) => {
    const artifactSummary = (run.compiler_artifact_summary && typeof run.compiler_artifact_summary === 'object')
      ? run.compiler_artifact_summary
      : null;
    if (!artifactSummary) {
      toast.error('Compiler artifact context is unavailable for this validation run');
      return;
    }
    if (action === 'create_regression_explanation') {
      const experimentRunIds = Array.isArray(artifactSummary.source_run_ids) ? artifactSummary.source_run_ids.filter(Boolean) : [];
      const primaryRunId = String(artifactSummary.primary_run_id || '').trim();
      const comparisonRunId = String(artifactSummary.comparison_run_id || '').trim();
      if (experimentRunIds.length !== 2 || !primaryRunId || !comparisonRunId) {
        toast.error('Explanation generation requires two compared validation runs');
        return;
      }
      createCompilerArtifactMutation.mutate({
        job_type: 'compiler_regression_explanation',
        title: `${String(run.name || 'Compiler Validation').trim()} Explanation`,
        document_ids: [],
        experiment_run_ids: experimentRunIds,
        primary_run_id: primaryRunId,
        comparison_run_id: comparisonRunId,
        output_format: 'markdown',
        output_style: 'technical',
      });
      return;
    }
    if (action === 'create_patch_proposal') {
      const noteId = String(artifactSummary.explanation_note_id || '').trim();
      if (!noteId) {
        toast.error('Patch proposal generation requires an explanation note');
        return;
      }
      createCompilerArtifactMutation.mutate({
        job_type: 'compiler_patch_proposal',
        title: `${String(run.name || 'Compiler Validation').trim()} Patch Proposal`,
        document_ids: [],
        research_note_id: noteId,
        output_format: 'markdown',
        output_style: 'technical',
      });
      return;
    }
    const noteId = String(artifactSummary.proposal_note_id || '').trim();
    const explicitSourceId = String(artifactSummary.source_id || '').trim();
    const profileRepoSourceIds = Array.isArray(ownerProfile?.repo_source_ids) ? ownerProfile?.repo_source_ids.filter(Boolean) : [];
    const sourceId = explicitSourceId || (profileRepoSourceIds.length === 1 ? String(profileRepoSourceIds[0]) : '');
    if (!noteId) {
      toast.error('Patch draft generation requires a proposal note');
      return;
    }
    if (!sourceId) {
      toast.error('Patch draft generation requires one repo source');
      return;
    }
    createCompilerArtifactMutation.mutate({
      job_type: 'compiler_patch_draft',
      title: `${String(run.name || 'Compiler Validation').trim()} Patch Draft`,
      document_ids: [],
      research_note_id: noteId,
      source_id: sourceId,
      output_format: 'markdown',
      output_style: 'technical',
    });
  }, [createCompilerArtifactMutation]);
  const renderScientificValidationRuns = useCallback(
    (
      runs?: Array<Record<string, any>> | null,
      options?: {
        ownerProfile?: DomainResearchProfile | null;
      }
    ) => {
      if (!Array.isArray(runs) || runs.length === 0) {
        return <div className="text-gray-500">No recent validation runs.</div>;
      }
      return (
        <div className="space-y-2">
          {runs.slice(0, 5).map((run) => {
            const status = String(run.status || 'unknown');
            const blockedReason = String(run.blocked_reason_code || '').trim();
            const latestOperatorAction = String(run.latest_operator_action || '').trim();
            const latestOperatorOutcome = String(run.latest_operator_outcome_status || '').trim();
            const sandboxName =
              String(run.sandbox_profile_name || '').trim() ||
              String(scientificSandboxProfileById[String(run.sandbox_profile_id || '')]?.name || '').trim() ||
              String(run.sandbox_profile_id || '').trim() ||
              'default sandbox';
            const typedRun = run as ScientificValidationRunSummary;
            const resolvedOwnerProfile =
              options?.ownerProfile
              || domainProfileById[String(typedRun.domain_research_profile_id || '')]
              || null;
            const artifactSummary = (typedRun.compiler_artifact_summary && typeof typedRun.compiler_artifact_summary === 'object')
              ? typedRun.compiler_artifact_summary
              : null;
            const availableActions = Array.isArray(artifactSummary?.available_actions) ? artifactSummary?.available_actions : [];
            const showPatchDraftAction = availableActions.includes('create_patch_draft')
              && (
                String(artifactSummary?.source_id || '').trim()
                || (Array.isArray(resolvedOwnerProfile?.repo_source_ids) && resolvedOwnerProfile?.repo_source_ids.length === 1)
              );
            return (
              <div key={String(run.id || run.name)} className="border border-gray-100 rounded p-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="font-medium text-gray-900">{String(run.name || run.id)}</div>
                  <span className={`text-[11px] px-2 py-0.5 rounded ${scientificValidationStatusClasses(status)}`}>
                    {status}
                  </span>
                </div>
                <div className="mt-1 text-gray-600">
                  {run.recipe_family ? `Recipe ${String(run.recipe_family)}` : 'Scientific validation'}
                  {run.recipe_id ? ` · ${String(run.recipe_id)}` : ''}
                  {sandboxName ? ` · Sandbox ${sandboxName}` : ''}
                </div>
                <div className="mt-1 text-gray-500">
                  Progress {Number(run.progress || 0)}%
                  {run.completed_at ? ` · Completed ${new Date(String(run.completed_at)).toLocaleString()}` : ''}
                  {!run.completed_at && run.created_at ? ` · Created ${new Date(String(run.created_at)).toLocaleString()}` : ''}
                </div>
                {latestOperatorAction ? (
                  <div className="mt-1 text-slate-600">
                    Latest action: {latestOperatorAction}
                    {latestOperatorOutcome ? ` · ${latestOperatorOutcome}` : ''}
                  </div>
                ) : null}
                {Number(run.retry_count || 0) > 0 || run.parent_run_id || run.latest_child_run_id ? (
                  <div className="mt-1 text-slate-500">
                    Retry lineage
                    {Number(run.retry_count || 0) > 0 ? ` · attempt ${Number(run.retry_count || 0)}` : ''}
                    {run.parent_run_id ? ` · parent ${String(run.parent_run_id)}` : ''}
                    {run.latest_child_run_id ? ` · child ${String(run.latest_child_run_id)}` : ''}
                  </div>
                ) : null}
                {blockedReason ? (
                  <div className="mt-1 text-rose-700">Blocked: {humanizeScientificValidationReason(blockedReason)}</div>
                ) : null}
                {artifactSummary ? (
                  <div className="mt-2 rounded border border-indigo-100 bg-indigo-50 p-2">
                    <div className="text-[11px] font-medium text-indigo-900">Compiler artifact handoff</div>
                    <div className="mt-1 flex flex-wrap gap-2 text-[11px]">
                      {artifactSummary.explanation_note_id ? (
                        <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-700">Explanation ready</span>
                      ) : artifactSummary.explanation_synthesis_job_id ? (
                        <span className={`px-2 py-0.5 rounded ${synthesisStatusClasses(artifactSummary.explanation_synthesis_status)}`}>
                          Explanation {String(artifactSummary.explanation_synthesis_status || 'pending').replace(/_/g, ' ')}
                        </span>
                      ) : null}
                      {artifactSummary.proposal_note_id ? (
                        <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-700">Proposal ready</span>
                      ) : artifactSummary.proposal_synthesis_job_id ? (
                        <span className={`px-2 py-0.5 rounded ${synthesisStatusClasses(artifactSummary.proposal_synthesis_status)}`}>
                          Proposal {String(artifactSummary.proposal_synthesis_status || 'pending').replace(/_/g, ' ')}
                        </span>
                      ) : null}
                      {artifactSummary.patch_draft_note_id ? (
                        <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-700">Patch draft ready</span>
                      ) : artifactSummary.patch_draft_synthesis_job_id ? (
                        <span className={`px-2 py-0.5 rounded ${synthesisStatusClasses(artifactSummary.patch_draft_synthesis_status)}`}>
                          Patch draft {String(artifactSummary.patch_draft_synthesis_status || 'pending').replace(/_/g, ' ')}
                        </span>
                      ) : null}
                    </div>
                    {artifactSummary.source_run_ids?.length ? (
                      <div className="mt-1 text-[11px] text-indigo-800">
                        Run pair: {artifactSummary.source_run_ids.join(' vs ')}
                      </div>
                    ) : null}
                    {artifactSummary.source_id || artifactSummary.source_name ? (
                      <div className="mt-1 text-[11px] text-indigo-800">
                        Repo source: {String(artifactSummary.source_name || artifactSummary.source_id)}
                      </div>
                    ) : null}
                    <div className="mt-2 flex flex-wrap gap-2">
                      {availableActions.includes('create_regression_explanation') ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={createCompilerArtifactMutation.isLoading}
                          onClick={() => runCompilerArtifactAction(typedRun, 'create_regression_explanation', resolvedOwnerProfile)}
                        >
                          Create explanation
                        </Button>
                      ) : null}
                      {availableActions.includes('create_patch_proposal') ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={createCompilerArtifactMutation.isLoading}
                          onClick={() => runCompilerArtifactAction(typedRun, 'create_patch_proposal', resolvedOwnerProfile)}
                        >
                          Create proposal
                        </Button>
                      ) : null}
                      {showPatchDraftAction ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={createCompilerArtifactMutation.isLoading}
                          onClick={() => runCompilerArtifactAction(typedRun, 'create_patch_draft', resolvedOwnerProfile)}
                        >
                          Create patch draft
                        </Button>
                      ) : null}
                      {artifactSummary.explanation_note_id ? (
                        <Button size="sm" variant="ghost" onClick={() => openResearchNote(artifactSummary.explanation_note_id)}>
                          Open explanation note
                        </Button>
                      ) : null}
                      {artifactSummary.proposal_note_id ? (
                        <Button size="sm" variant="ghost" onClick={() => openResearchNote(artifactSummary.proposal_note_id)}>
                          Open proposal note
                        </Button>
                      ) : null}
                      {artifactSummary.patch_draft_note_id ? (
                        <Button size="sm" variant="ghost" onClick={() => openResearchNote(artifactSummary.patch_draft_note_id)}>
                          Open patch draft note
                        </Button>
                      ) : null}
                      {!artifactSummary.explanation_note_id && artifactSummary.explanation_synthesis_job_id && String(artifactSummary.explanation_synthesis_status || '').trim().toLowerCase() === 'completed' ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={saveCompilerArtifactNoteMutation.isLoading}
                          onClick={() => saveCompilerArtifactNoteMutation.mutate({ jobId: String(artifactSummary.explanation_synthesis_job_id) })}
                        >
                          Save explanation note
                        </Button>
                      ) : null}
                      {!artifactSummary.proposal_note_id && artifactSummary.proposal_synthesis_job_id && String(artifactSummary.proposal_synthesis_status || '').trim().toLowerCase() === 'completed' ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={saveCompilerArtifactNoteMutation.isLoading}
                          onClick={() => saveCompilerArtifactNoteMutation.mutate({ jobId: String(artifactSummary.proposal_synthesis_job_id) })}
                        >
                          Save proposal note
                        </Button>
                      ) : null}
                      {!artifactSummary.patch_draft_note_id && artifactSummary.patch_draft_synthesis_job_id && String(artifactSummary.patch_draft_synthesis_status || '').trim().toLowerCase() === 'completed' ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={saveCompilerArtifactNoteMutation.isLoading}
                          onClick={() => saveCompilerArtifactNoteMutation.mutate({ jobId: String(artifactSummary.patch_draft_synthesis_job_id) })}
                        >
                          Save patch draft note
                        </Button>
                      ) : null}
                      {run.agent_job_id ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => navigate(buildAutonomousAgentsUrl(String(run.agent_job_id)))}
                        >
                          Open source job
                        </Button>
                      ) : null}
                    </div>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      );
    },
    [buildAutonomousAgentsUrl, createCompilerArtifactMutation.isLoading, domainProfileById, navigate, openResearchNote, runCompilerArtifactAction, saveCompilerArtifactNoteMutation, scientificSandboxProfileById]
  );

  const filteredMonitorAnalytics = useMemo(() => {
    const monitors = ((monitorAnalyticsData as ResearchMonitorAnalyticsResponse | undefined)?.monitors || []).filter((monitor) => {
      if (healthCustomerFilter && String(monitor.customer || '') !== healthCustomerFilter) {
        return false;
      }
      if (healthMonitorTypeFilter && String(monitor.monitor_job_type || '') !== healthMonitorTypeFilter) {
        return false;
      }
      if (healthBucketFilter && String(monitor.health_bucket || '') !== healthBucketFilter) {
        return false;
      }
      if (
        healthAutonomyFilter &&
        !Object.entries(monitor.policy_mode_counts || {}).some(
          ([key, value]) => key === healthAutonomyFilter && Number(value || 0) > 0
        )
      ) {
        return false;
      }
      return true;
    });

    const totals = monitors.reduce(
      (acc, monitor) => {
        acc.total_monitors += 1;
        acc.discovered_count += monitor.discovered_count || 0;
        acc.accepted_count += monitor.accepted_count || 0;
        acc.rejected_count += monitor.rejected_count || 0;
        acc.auto_launched_count += monitor.auto_launched_count || 0;
        acc.approval_launched_count += monitor.approval_launched_count || 0;
        acc.blocked_count += monitor.blocked_count || 0;
        acc.follow_up_completed_count += monitor.follow_up_completed_count || 0;
        acc.follow_up_failed_count += monitor.follow_up_failed_count || 0;
        acc.follow_up_cancelled_count += monitor.follow_up_cancelled_count || 0;
        if (monitor.health_bucket === 'strong') {
          acc.strong_monitors += 1;
        } else if (monitor.health_bucket === 'mixed') {
          acc.mixed_monitors += 1;
        } else {
          acc.weak_monitors += 1;
        }
        return acc;
      },
      {
        total_monitors: 0,
        discovered_count: 0,
        accepted_count: 0,
        rejected_count: 0,
        auto_launched_count: 0,
        approval_launched_count: 0,
        blocked_count: 0,
        follow_up_completed_count: 0,
        follow_up_failed_count: 0,
        follow_up_cancelled_count: 0,
        strong_monitors: 0,
        mixed_monitors: 0,
        weak_monitors: 0,
      }
    );

    const recommendationMap = new Map<string, any>();
    monitors.forEach((monitor) => {
      (monitor.top_recommendations || []).forEach((recommendation) => {
        const current = recommendationMap.get(recommendation.recommendation_key) || {
          recommendation_key: recommendation.recommendation_key,
          launch_count: 0,
          auto_launch_count: 0,
          approval_launch_count: 0,
          blocked_count: 0,
          completed_count: 0,
          failed_count: 0,
          cancelled_count: 0,
          success_rate: 0,
          score_trend: 'mixed',
          monitor_count: 0,
        };
        current.launch_count += recommendation.launch_count || 0;
        current.auto_launch_count += recommendation.auto_launch_count || 0;
        current.approval_launch_count += recommendation.approval_launch_count || 0;
        current.blocked_count += recommendation.blocked_count || 0;
        current.completed_count += recommendation.completed_count || 0;
        current.failed_count += recommendation.failed_count || 0;
        current.cancelled_count += recommendation.cancelled_count || 0;
        current.monitor_count += 1;
        recommendationMap.set(recommendation.recommendation_key, current);
      });
    });

    const recommendations = Array.from(recommendationMap.values())
      .map((recommendation) => {
        const terminal =
          recommendation.completed_count + recommendation.failed_count + recommendation.cancelled_count;
        const successRate = terminal > 0 ? Number(((recommendation.completed_count / terminal) * 100).toFixed(1)) : 0;
        return {
          ...recommendation,
          success_rate: successRate,
          score_trend:
            recommendation.completed_count > recommendation.failed_count + recommendation.cancelled_count
              ? 'positive'
              : recommendation.failed_count + recommendation.cancelled_count > recommendation.completed_count
                ? 'negative'
                : 'mixed',
        };
      })
      .sort((a, b) => b.completed_count - a.completed_count || b.launch_count - a.launch_count || a.recommendation_key.localeCompare(b.recommendation_key))
      .slice(0, 6);

    return { monitors, totals, recommendations };
  }, [monitorAnalyticsData, healthAutonomyFilter, healthBucketFilter, healthCustomerFilter, healthMonitorTypeFilter]);

  useEffect(() => {
    const normalizedMonitorJobId = String(deepLinkedHealthMonitor || '').trim();
    if (!deepLinkedHealthTab || !normalizedMonitorJobId) return;
    const targetNode = healthMonitorCardRefs.current[normalizedMonitorJobId];
    if (!targetNode) return;
    targetNode.scrollIntoView?.({ behavior: 'smooth', block: 'center' });
  }, [deepLinkedHealthMonitor, deepLinkedHealthTab, filteredMonitorAnalytics.monitors]);

  const getHealthPolicyDraft = useCallback((monitor: ResearchMonitorHealthSummary) => {
    const key = String(monitor.monitor_job_id || '');
    if (key && healthPolicyDrafts[key]) {
      return healthPolicyDrafts[key];
    }
    return {
      automation_profile: String(monitor.automation_profile || monitor.autonomy_mode || 'balanced').trim().toLowerCase() === 'max_autonomy' ? 'max_autonomy' : 'balanced',
      mode: canonicalReviewModeFromMonitor(monitor),
      allowed: canonicalAllowedRecommendationsFromMonitor(monitor),
    };
  }, [healthPolicyDrafts]);

  const setHealthPolicyDraft = useCallback((monitorJobId: string, nextDraft: { automation_profile: string; mode: string; allowed: string[] }) => {
    setHealthPolicyDrafts((prev) => ({ ...prev, [monitorJobId]: nextDraft }));
  }, []);

  const getHealthBudgetDraft = useCallback((monitor: ResearchMonitorHealthSummary) => {
    const key = String(monitor.monitor_job_id || '');
    if (key && healthBudgetDrafts[key]) {
      return healthBudgetDrafts[key];
    }
    return {
      auto_launch_limit_24h: Number(monitor.autonomy_budget?.auto_launch_limit_24h || 0),
      approval_queue_limit_24h: Number(monitor.autonomy_budget?.approval_queue_limit_24h || 0),
      alert_limit_24h: Number(monitor.autonomy_budget?.alert_limit_24h || 0),
      queue_backlog_cap: Number(monitor.autonomy_budget?.queue_backlog_cap || 0),
    };
  }, [healthBudgetDrafts]);

  const setHealthBudgetDraft = useCallback(
    (
      monitorJobId: string,
      nextDraft: { auto_launch_limit_24h: number; approval_queue_limit_24h: number; alert_limit_24h: number; queue_backlog_cap: number }
    ) => {
      setHealthBudgetDrafts((prev) => ({ ...prev, [monitorJobId]: nextDraft }));
    },
    []
  );

  const getHealthCustomerBudgetDraft = useCallback((customerRow: any) => {
    const key = String(customerRow.customer || '');
    if (key && healthCustomerBudgetDrafts[key]) {
      return healthCustomerBudgetDrafts[key];
    }
    return {
      auto_launch_limit_24h: Number(customerRow.customer_budget?.auto_launch_limit_24h || 0),
      approval_queue_limit_24h: Number(customerRow.customer_budget?.approval_queue_limit_24h || 0),
      alert_limit_24h: Number(customerRow.customer_budget?.alert_limit_24h || 0),
      queue_backlog_cap: Number(customerRow.customer_budget?.queue_backlog_cap || 0),
    };
  }, [healthCustomerBudgetDrafts]);

  const setHealthCustomerBudgetDraft = useCallback(
    (
      customer: string,
      nextDraft: { auto_launch_limit_24h: number; approval_queue_limit_24h: number; alert_limit_24h: number; queue_backlog_cap: number }
    ) => {
      setHealthCustomerBudgetDrafts((prev) => ({ ...prev, [customer]: nextDraft }));
    },
    []
  );

  const getHealthCustomerRebalancePreview = useCallback((customer: string) => {
    return healthCustomerRebalancePreviews[String(customer || '').trim()];
  }, [healthCustomerRebalancePreviews]);

  const buildCustomerRebalanceUpdates = useCallback((customerRow: ResearchMonitorCustomerPortfolio) => {
    return (customerRow.rebalance_guidance_changes || []).map((change) => ({
      monitor_job_id: String(change.monitor_job_id),
      auto_launch_limit_24h: Number(change.proposed_budget?.auto_launch_limit_24h || 0),
      approval_queue_limit_24h: Number(change.proposed_budget?.approval_queue_limit_24h || 0),
      alert_limit_24h: Number(change.proposed_budget?.alert_limit_24h || 0),
      queue_backlog_cap: Number(change.proposed_budget?.queue_backlog_cap || 0),
    }));
  }, []);

  const getHealthPolicyAnalyticsContext = useCallback((monitor: ResearchMonitorHealthSummary) => ({
    health_bucket: monitor.health_bucket,
    policy_confidence: monitor.policy_confidence,
    accepted_count: monitor.accepted_count,
    blocked_count: monitor.blocked_count,
    follow_up_completed_count: monitor.follow_up_completed_count,
    follow_up_failed_count: monitor.follow_up_failed_count,
    follow_up_cancelled_count: monitor.follow_up_cancelled_count,
  }), []);

  const formatPolicyHistoryTimestamp = useCallback((value?: string) => {
    if (!value) return 'Unknown time';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return value;
    }
    return date.toLocaleString();
  }, []);

  const formatPolicyChangeSource = useCallback((value?: string) => {
    const source = String(value || '').trim() || 'manual_override';
    return source.replace(/_/g, ' ');
  }, []);

  const formatSimulationCountDelta = useCallback((value: number) => {
    if (value > 0) return `+${value}`;
    return String(value);
  }, []);

  const formatPolicyEvaluationStatus = useCallback((value?: string) => {
    const normalized = String(value || '').trim().toLowerCase();
    if (!normalized) return 'Unknown';
    if (normalized === 'insufficient_data') return 'Insufficient data';
    return normalized.replace(/_/g, ' ');
  }, []);

  const openInboxForMonitorSignal = useCallback(
    (monitorJobId: string, inboxItemId?: string, policyDrilldown?: InboxPolicyDrilldown) => {
      setActiveTab('inbox');
      setInboxStatusFilter('accepted');
      setInboxTypeFilter('');
      setInboxSearch('');
      setInboxCustomerFilter('');
      setInboxJobFilter(String(monitorJobId || '').trim());
      setInboxHealthDrilldown('');
      setInboxPolicyDrilldown(policyDrilldown || '');
      const params = new URLSearchParams(location.search);
      params.set('tab', 'inbox');
      params.set('inbox_job', String(monitorJobId || '').trim());
      params.delete('inbox_customer');
      params.delete('inbox_health_drilldown');
      if (inboxItemId && String(inboxItemId).trim()) {
        params.set('inbox', String(inboxItemId).trim());
      } else {
        params.delete('inbox');
      }
      if (policyDrilldown && String(policyDrilldown).trim()) {
        params.set('inbox_policy_drilldown', String(policyDrilldown).trim());
      } else {
        params.delete('inbox_policy_drilldown');
      }
      params.delete('job');
      navigate(`${location.pathname}?${params.toString()}`, { replace: true });
    },
    [location.pathname, location.search, navigate]
  );

  // View chain status
  const viewChainStatus = async (jobId: string) => {
    try {
      const status = await apiClient.getChainStatus(jobId);
      setSelectedChainStatus(status);
    } catch (error) {
      console.error('Failed to load chain status:', error);
      toast.error('Failed to load chain status');
    }
  };

  const chainExperimentStopInfo = useMemo(() => {
    const cs = selectedChainStatus;
    if (!cs || !Array.isArray(cs.jobs)) return null;

    let found: any = null;
    for (let i = cs.jobs.length - 1; i >= 0; i--) {
      const job: any = cs.jobs[i];
      const results = job?.results;
      const stop = results?.experiment_loop_stop;
      if (stop && typeof stop === 'object') {
        found = { stop, job };
        break;
      }
    }
    if (!found) return null;

    let noteId: string | null = null;
    for (const j of cs.jobs as any[]) {
      const cfg = j?.config;
      const id = String(cfg?.research_note_id || cfg?.note_id || '').trim();
      if (id) {
        noteId = id;
        break;
      }
    }

    return {
      reason: String(found.stop?.reason || '').trim(),
      atRunId: String(found.stop?.at_run_id || '').trim(),
      stoppedByJobId: String(found.job?.id || '').trim(),
      noteId,
    };
  }, [selectedChainStatus]);

  // Mutations
  const actionMutation = useMutation(
    ({
      jobId,
      action,
      checkpointNote,
      checkpointActionPatch,
      actionPayload,
    }: {
      jobId: string;
      action:
        | 'pause'
        | 'resume'
        | 'cancel'
        | 'restart'
        | 'relaunch'
        | 'launch_tie_breaker'
        | 'promote_swarm_candidate'
        | 'assign_swarm_review'
        | 'clear_swarm_assignment'
        | 'update_swarm_review_note'
        | 'approve'
        | 'reject'
        | 'edit'
        | 'skip';
      checkpointNote?: string;
      checkpointActionPatch?: Record<string, any>;
      actionPayload?: Record<string, any>;
    }) =>
      apiClient.performAgentJobAction(jobId, action, {
        checkpoint_note: checkpointNote,
        checkpoint_action_patch: checkpointActionPatch,
        action_payload: actionPayload,
      }),
    {
      onSuccess: (job, vars) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        if (
          (vars?.action === 'relaunch')
          || (vars?.action === 'restart' && String(job?.id || '') !== String(vars?.jobId || ''))
          || vars?.action === 'launch_tie_breaker'
          || vars?.action === 'promote_swarm_candidate'
        ) {
          const successMessage =
            vars?.action === 'restart'
              ? 'Refined retry started as a new job'
              : vars?.action === 'launch_tie_breaker'
                ? 'Verifier tie-breaker started'
                : vars?.action === 'promote_swarm_candidate'
                  ? 'Repair chain launched from swarm candidate'
                  : 'Relaunched as a new job';
          toast.success(successMessage);
          setSelectedJob(job);
          navigate(buildAutonomousAgentsUrl(String(job.id)), { replace: true });
          return;
        }
        if (vars?.action === 'update_swarm_review_note') {
          setSwarmReviewNoteDrafts((prev) => {
            const next = { ...prev };
            delete next[String(vars.jobId || '')];
            return next;
          });
          toast.success('Swarm note saved');
        } else if (vars?.action === 'assign_swarm_review') {
          toast.success('Swarm review assignment updated');
        } else if (vars?.action === 'clear_swarm_assignment') {
          toast.success('Swarm review assignment cleared');
        }
        if (vars?.action === 'update_swarm_review_note' || vars?.action === 'assign_swarm_review' || vars?.action === 'clear_swarm_assignment') {
          if (selectedJob?.id === job.id) setSelectedJob(job);
          return;
        }
        if (vars?.action === 'approve') {
          toast.success('Checkpoint approved');
        } else if (vars?.action === 'edit') {
          toast.success('Checkpoint edited and approved');
        } else if (vars?.action === 'skip') {
          toast.success('Step skipped and resumed');
        } else if (vars?.action === 'reject') {
          toast.success('Checkpoint rejected');
        } else {
          toast.success(`Job ${job.status}`);
        }
        if (selectedJob?.id === job.id) setSelectedJob(job);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Action failed');
      },
    }
  );

  const promoteDomainResearchMutation = useMutation(
    ({
      jobId,
      data,
    }: {
      jobId: string;
      data: AgentJobPromoteDomainResearchRequest;
    }) => apiClient.promoteDomainResearchAgentJob(jobId, data),
    {
      onSuccess: (response: AgentJobPromoteDomainResearchResponse) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['research-portfolios']);
        if (response?.source_job) setSelectedJob(response.source_job);
        toast.success(
          response?.research_portfolio_id
            ? 'Promoted to monitor and fleet'
            : 'Promoted to monitor'
        );
      },
      onError: (error: any) => {
        toast.error(error.message || 'Promotion failed');
      },
    }
  );

  const deleteMutation = useMutation(
    (jobId: string) => apiClient.deleteAgentJob(jobId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        toast.success('Job deleted');
        setSelectedJob(null);
        navigate(buildAutonomousAgentsUrl(), { replace: true });
      },
      onError: (error: any) => {
        toast.error(error.message || 'Delete failed');
      },
    }
  );

  const updateInboxItemMutation = useMutation(
    ({ itemId, data }: { itemId: string; data: { status?: ResearchInboxItemStatus; feedback?: string; metadata_patch?: Record<string, any> } }) =>
      apiClient.updateResearchInboxItem(itemId, data),
    {
      onSuccess: (_res, vars) => {
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        if (vars?.data?.status === 'rejected') {
          setInboxRejectReasonDrafts((current) => {
            const next = { ...current };
            delete next[String(vars.itemId || '')];
            return next;
          });
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Update failed');
      },
    }
  );

  const bulkUpdateInboxMutation = useMutation(
    ({ itemIds, data }: { itemIds: string[]; data: { status?: ResearchInboxItemStatus; feedback?: string } }) =>
      apiClient.bulkUpdateResearchInboxItems({ item_ids: itemIds, ...data }),
    {
      onSuccess: (res) => {
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        setSelectedInboxIds({});
        setInboxBulkRejectReason('');
        toast.success(`Updated ${res.updated} items`);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Bulk update failed');
      },
    }
  );

  const upsertMonitorProfileMutation = useMutation(
    (data: { customer?: string; muted_tokens?: string[]; muted_patterns?: string[]; notes?: string; merge_lists?: boolean }) =>
      apiClient.upsertResearchMonitorProfile(data),
    {
      onSuccess: () => {
        toast.success('Monitor profile updated');
        queryClient.invalidateQueries(['research-monitor-profiles']);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update monitor profile');
      },
    }
  );

  const updateMonitorPolicyMutation = useMutation(
    ({
      monitorJobId,
      data,
    }: {
      monitorJobId: string;
      data: {
        automation_profile?: string;
        automation_policy?: Record<string, any>;
        mode?: string;
        allowed_recommendations?: string[];
        reset_to_default?: boolean;
        change_source?: string;
        change_reason?: string;
        analytics_context?: Record<string, any>;
      };
    }) =>
      apiClient.updateResearchMonitorPolicy(monitorJobId, data),
    {
      onSuccess: (_res, vars) => {
        queryClient.invalidateQueries(['research-monitor-analytics']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['research-inbox']);
        toast.success('Monitor policy updated');
        if (vars.monitorJobId) {
          setHealthPolicySimulations((prev) => {
            const next = { ...prev };
            delete next[vars.monitorJobId];
            return next;
          });
          setHealthPolicyEvaluations((prev) =>
            Object.fromEntries(Object.entries(prev).filter(([key]) => !key.startsWith(`${vars.monitorJobId}:`)))
          );
          setHealthPolicyDrafts((prev) => {
            const next = { ...prev };
            delete next[vars.monitorJobId];
            return next;
          });
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update monitor policy');
      },
    }
  );

  const rollbackMonitorPolicyMutation = useMutation(
    ({ monitorJobId, historyEntryId }: { monitorJobId: string; historyEntryId: string }) =>
      apiClient.rollbackResearchMonitorPolicy(monitorJobId, { history_entry_id: historyEntryId }),
    {
      onSuccess: (_res, vars) => {
        queryClient.invalidateQueries(['research-monitor-analytics']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['research-inbox']);
        toast.success('Monitor policy rolled back');
        if (vars.monitorJobId) {
          setHealthPolicySimulations((prev) => {
            const next = { ...prev };
            delete next[vars.monitorJobId];
            return next;
          });
          setHealthPolicyEvaluations((prev) =>
            Object.fromEntries(Object.entries(prev).filter(([key]) => !key.startsWith(`${vars.monitorJobId}:`)))
          );
          setHealthPolicyDrafts((prev) => {
            const next = { ...prev };
            delete next[vars.monitorJobId];
            return next;
          });
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to roll back monitor policy');
      },
    }
  );

  const updateMonitorBudgetMutation = useMutation(
    ({
      monitorJobId,
      data,
    }: {
      monitorJobId: string;
      data: {
        auto_launch_limit_24h?: number;
        approval_queue_limit_24h?: number;
        alert_limit_24h?: number;
        queue_backlog_cap?: number;
        reset_to_default?: boolean;
      };
    }) => apiClient.updateResearchMonitorBudget(monitorJobId, data),
    {
      onSuccess: (_res, vars) => {
        queryClient.invalidateQueries(['research-monitor-analytics']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        toast.success('Monitor autonomy budget updated');
        if (vars.monitorJobId) {
          setHealthBudgetDrafts((prev) => {
            const next = { ...prev };
            delete next[vars.monitorJobId];
            return next;
          });
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update monitor budget');
      },
    }
  );

  const updateCustomerBudgetMutation = useMutation(
    ({
      customer,
      data,
    }: {
      customer: string;
      data: {
        auto_launch_limit_24h?: number;
        approval_queue_limit_24h?: number;
        alert_limit_24h?: number;
        queue_backlog_cap?: number;
        reset_to_default?: boolean;
      };
    }) => apiClient.updateResearchMonitorCustomerBudget({ customer, ...data }),
    {
      onSuccess: (_res, vars) => {
        queryClient.invalidateQueries(['research-monitor-analytics']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['research-inbox']);
        toast.success('Customer autonomy budget updated');
        if (vars.customer) {
          setHealthCustomerBudgetDrafts((prev) => {
            const next = { ...prev };
            delete next[vars.customer];
            return next;
          });
          setHealthCustomerRebalancePreviews((prev) => {
            const next = { ...prev };
            delete next[vars.customer];
            return next;
          });
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update customer budget');
      },
    }
  );

  const previewCustomerRebalanceMutation = useMutation(
    ({
      customer,
      monitorBudgetUpdates,
    }: {
      customer: string;
      monitorBudgetUpdates?: Array<{
        monitor_job_id: string;
        auto_launch_limit_24h: number;
        approval_queue_limit_24h: number;
        alert_limit_24h: number;
        queue_backlog_cap: number;
      }>;
    }) => apiClient.previewResearchMonitorCustomerRebalance({ customer, monitor_budget_updates: monitorBudgetUpdates }),
    {
      onSuccess: (result) => {
        setHealthCustomerRebalancePreviews((prev) => ({
          ...prev,
          [result.customer]: result,
        }));
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to preview customer rebalance');
      },
    }
  );

  const applyCustomerRebalanceMutation = useMutation(
    ({
      customer,
      monitorBudgetUpdates,
      changeReason,
    }: {
      customer: string;
      monitorBudgetUpdates: Array<{
        monitor_job_id: string;
        auto_launch_limit_24h: number;
        approval_queue_limit_24h: number;
        alert_limit_24h: number;
        queue_backlog_cap: number;
      }>;
      changeReason?: string;
    }) =>
      apiClient.applyResearchMonitorCustomerRebalance({
        customer,
        monitor_budget_updates: monitorBudgetUpdates,
        change_reason: changeReason,
      }),
    {
      onSuccess: (_res, vars) => {
        queryClient.invalidateQueries(['research-monitor-analytics']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['research-inbox']);
        toast.success('Customer rebalance applied');
        setHealthCustomerRebalancePreviews((prev) => {
          const next = { ...prev };
          delete next[vars.customer];
          return next;
        });
        vars.monitorBudgetUpdates.forEach((row) => {
          setHealthBudgetDrafts((prev) => {
            const next = { ...prev };
            delete next[row.monitor_job_id];
            return next;
          });
        });
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to apply customer rebalance');
      },
    }
  );

  const simulateMonitorPolicyMutation = useMutation(
    ({
      monitorJobId,
      data,
    }: {
      monitorJobId: string;
      data: {
        automation_profile?: string;
        automation_policy?: Record<string, any>;
        mode?: string;
        allowed_recommendations?: string[];
        history_limit?: number;
      };
    }) => apiClient.simulateResearchMonitorPolicy(monitorJobId, data),
    {
      onSuccess: (result) => {
        setHealthPolicySimulations((prev) => ({
          ...prev,
          [String(result.monitor_job_id)]: result,
        }));
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to preview policy impact');
      },
    }
  );

  const loadPolicyEvaluationMutation = useMutation(
    ({ monitorJobId, historyEntryId }: { monitorJobId: string; historyEntryId: string }) =>
      apiClient.getResearchMonitorPolicyEvaluation(monitorJobId, historyEntryId),
    {
      onSuccess: (result) => {
        setHealthPolicyEvaluations((prev) => ({
          ...prev,
          [`${result.monitor_job_id}:${result.history_entry_id}`]: result,
        }));
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to load policy comparison');
      },
    }
  );

  const loadCustomerRebalanceEvaluationMutation = useMutation(
    ({ customer, historyEntryId }: { customer: string; historyEntryId: string }) =>
      apiClient.getResearchMonitorCustomerRebalanceEvaluation(customer, historyEntryId),
    {
      onSuccess: (result) => {
        setHealthCustomerRebalanceEvaluations((prev) => ({
          ...prev,
          [`${result.customer}:${result.history_entry_id}`]: result,
        }));
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to load rebalance comparison');
      },
    }
  );

  const openHealthPolicyComparison = useCallback((monitorJobId: string, historyEntryId?: string) => {
    setActiveTab('health');
    if (monitorJobId && historyEntryId) {
      loadPolicyEvaluationMutation.mutate({ monitorJobId, historyEntryId });
    }
  }, [loadPolicyEvaluationMutation]);

  useEffect(() => {
    const monitorJobId = String(deepLinkedHealthMonitor || '').trim();
    const historyEntryId = String(deepLinkedHealthPolicyHistory || '').trim();
    if (!deepLinkedHealthTab || !monitorJobId || !historyEntryId) return;
    openHealthPolicyComparison(monitorJobId, historyEntryId);
  }, [deepLinkedHealthMonitor, deepLinkedHealthPolicyHistory, deepLinkedHealthTab, openHealthPolicyComparison]);

  const extractReposMutation = useMutation(
    (itemId: string) => apiClient.extractReposForInboxItem(itemId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-inbox']);
        toast.success('Repo links extracted');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to extract repos');
      },
    }
  );

  const relaunchInboxFollowUpMutation = useMutation(
    ({ itemId, operatorNote }: { itemId: string; operatorNote?: string }) =>
      apiClient.relaunchInboxFollowUp(itemId, operatorNote ? { operator_note: operatorNote } : {}),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        toast.success('Follow-up relaunched');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to relaunch follow-up');
      },
    }
  );

  const ingestRepoMutation = useMutation(
    (payload: { provider: 'github' | 'gitlab'; repo: string }) =>
      apiClient.requestGitRepository({
        provider: payload.provider,
        repositories: [payload.repo],
        include_files: true,
        include_issues: false,
        include_pull_requests: false,
        include_wiki: false,
        incremental_files: true,
        use_gitignore: true,
        max_pages: 5,
        auto_sync: true,
      }),
    {
      onSuccess: (src) => {
        toast.success(`Repo ingestion started: ${src.name}`);
        // show in documents sources list
        navigate(`/documents`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error.message || 'Failed to ingest repo');
      },
    }
  );

  const runPaperRepoCodeAgent = async (item: ResearchInboxItem, chosenRepoOverride?: string) => {
    try {
      // Ensure repos are extracted
      let repos = (item.metadata as any)?.repos;
      if (!Array.isArray(repos) || repos.length === 0) {
        const res = await apiClient.extractReposForInboxItem(item.id);
        repos = res.repos;
      }
      const githubRepos = (repos || []).filter((r: any) => String(r?.provider) === 'github');
      if (githubRepos.length === 0) {
        toast.error('No GitHub repos found for this paper yet');
        return;
      }
      const defaultRepo = String(githubRepos[0].repo || '').trim();
      const chosenRepo = String(chosenRepoOverride || paperRepoSelectionDrafts[item.id] || defaultRepo).trim();
      if (!chosenRepo) {
        toast.error('Select a GitHub repo first');
        return;
      }

      const goal = `Implement or integrate the paper's repository changes relevant to our product. Start from the ingested repo and produce a minimal patch.\n\nPaper: ${item.title}\n\nAbstract:\n${item.summary || ''}`.slice(
        0,
        1600
      );

      const chains = ((chainsData as any)?.chains || []) as any[];
      const chain = chains.find((c: any) => c?.name === 'arxiv_repo_code_patch_chain');
      if (!chain?.id) {
        toast.error('Chain definition not found (arxiv_repo_code_patch_chain)');
        return;
      }

      createFromChainMutation.mutate({
        chain_definition_id: chain.id,
        name_prefix: `Paper→Repo→Code — ${new Date().toLocaleDateString()}`,
        variables: {
          inbox_item_id: item.id,
          provider: 'github',
          repo: chosenRepo,
          goal,
        },
        config_overrides: {
          inbox_item_id: item.id,
          provider: 'github',
          repo: chosenRepo,
          // help code patch proposer pick relevant files
          search_query: `${item.title}\n${item.summary || ''}`.slice(0, 500),
        },
        start_immediately: true,
      });
      setActiveTab('jobs');
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to start chain');
    }
  };

  const runPaperAlgorithmProject = async (item: ResearchInboxItem, requestedBehavioralCheck: boolean, entrypoint: string) => {
    try {
      const chains = ((chainsData as any)?.chains || []) as any[];
      let preferredChainName = 'arxiv_algorithm_project_chain';
      let repos = (item.metadata as any)?.repos;
      if (!Array.isArray(repos) || repos.length === 0) {
        try {
          const res = await apiClient.extractReposForInboxItem(item.id);
          repos = res.repos;
        } catch {
          repos = repos || [];
        }
      }
      const hasGithubRepo = Array.isArray(repos) && repos.some((r: any) => String(r?.provider || '').toLowerCase() === 'github');
      if (hasGithubRepo) preferredChainName = 'arxiv_repo_algorithm_project_chain';

      const chain = chains.find((c: any) => c?.name === preferredChainName);
      if (!chain?.id) {
        toast.error(`Chain definition not found (${preferredChainName})`);
        return;
      }
      const allowBehavioral = !!requestedBehavioralCheck && unsafeExecBadge.status === 'ready';
      if (requestedBehavioralCheck && !allowBehavioral) {
        toast('Behavioral demo run requested, but server is not ready (see badge)');
      }
      const ep = String(entrypoint || 'demo.py').trim() || 'demo.py';
      createFromChainMutation.mutate({
        chain_definition_id: chain.id,
        name_prefix: `Paper→Algorithm — ${new Date().toLocaleDateString()}`,
        variables: { inbox_item_id: item.id },
        config_overrides: {
          inbox_item_id: item.id,
          language: 'python',
          include_tests: true,
          behavioral_check: allowBehavioral,
          entrypoint: ep,
        },
        start_immediately: true,
      });
      setActiveTab('jobs');
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to start algorithm implementation');
    }
  };

  const createMutation = useMutation(
    (data: AgentJobCreate) => apiClient.createAgentJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        toast.success('Job created');
        setShowCreateModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Create failed');
      },
    }
  );

  const createCodingBacklogMutation = useMutation(
    (data: CodingBacklogItemCreate) => apiClient.createCodingBacklogItem(data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['coding-backlog-items']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Coding backlog item created');
        setBacklogTitle('');
        setBacklogGoal('');
        setBacklogFailureSymptom('');
        setBacklogCommandsText('');
        setBacklogFilePathsText('');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create coding backlog item');
      },
    }
  );

  const createCodingSwarmProfileMutation = useMutation(
    (data: CodingSwarmProfileCreate) => apiClient.createCodingSwarmProfile(data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['coding-swarm-profiles']);
        toast.success('Coding swarm profile saved');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to save coding swarm profile');
      },
    }
  );

  const updateCodingSwarmProfileMutation = useMutation(
    ({ profileId, data }: { profileId: string; data: CodingSwarmProfileUpdate }) => apiClient.updateCodingSwarmProfile(profileId, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['coding-swarm-profiles']);
        toast.success('Coding swarm profile updated');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to update coding swarm profile');
      },
    }
  );

  const deleteCodingSwarmProfileMutation = useMutation(
    (profileId: string) => apiClient.deleteCodingSwarmProfile(profileId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['coding-swarm-profiles']);
        toast.success('Coding swarm profile deleted');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to delete coding swarm profile');
      },
    }
  );

  const openCodingSwarmProfileEditor = useCallback((profile?: CodingSwarmProfile | null, options?: { duplicate?: boolean }) => {
    const duplicate = Boolean(options?.duplicate);
    setEditingCodingSwarmProfileId(duplicate ? '' : String(profile?.id || ''));
    setCodingSwarmProfileDraft({
      title: duplicate
        ? `${String(profile?.title || 'Coding Swarm Profile').trim()} Copy`
        : String(profile?.title || '').trim(),
      source_id: String(profile?.source_id || codeSources[0]?.id || ''),
      preset_key: String(profile?.preset_key || 'bug_triage_swarm'),
      description: String(profile?.description || ''),
      scope_default: String(profile?.scope_default || 'auto'),
      default_commands: Array.isArray(profile?.default_commands) ? [...profile!.default_commands] : [],
      default_file_paths: Array.isArray(profile?.default_file_paths) ? [...profile!.default_file_paths] : [],
      max_agents: Math.max(1, Math.min(Number(profile?.max_agents || 4), 4)),
      safe_command_policy: String(profile?.safe_command_policy || 'standard'),
      saved_search_query: String(profile?.saved_search_query || ''),
      is_default: duplicate ? false : Boolean(profile?.is_default),
      status: String(profile?.status || 'active'),
      visibility: String(profile?.visibility || 'private'),
      shared_with_user_ids: Array.isArray(profile?.shared_with_user_ids) ? [...profile.shared_with_user_ids] : [],
      profile_metadata: (profile?.profile_metadata && typeof profile.profile_metadata === 'object') ? profile.profile_metadata : {},
      duplicate_mode: duplicate,
    });
    setActiveTab('profiles');
  }, [codeSources]);

  const closeCodingSwarmProfileEditor = useCallback(() => {
    setEditingCodingSwarmProfileId('');
    setCodingSwarmProfileDraft({
      title: '',
      source_id: '',
      preset_key: 'bug_triage_swarm',
      description: '',
      scope_default: 'auto',
      default_commands: [],
      default_file_paths: [],
      max_agents: 4,
      safe_command_policy: 'standard',
      saved_search_query: '',
      is_default: false,
      status: 'active',
      visibility: 'private',
      shared_with_user_ids: [],
      profile_metadata: {},
      duplicate_mode: false,
    });
  }, []);

  const createDomainProfileMutation = useMutation(
    (data: DomainResearchProfileCreate) => apiClient.createDomainResearchProfile(data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Domain profile created');
        setDomainProfileTitle('');
        setDomainProfileTopic('');
        setDomainProfileObjective('');
        setDomainProfileTrackType('compiler');
        setDomainProfileSourceScope('kb_plus_arxiv_plus_repo');
        setDomainProfileQueriesText('');
        setDomainProfileBenchmarkQueriesText('');
        setDomainProfileCadenceMinutes('1440');
        setDomainProfileRepoSelection({});
        setDomainProfileSandboxProfileId(resolveSandboxProfileId('compiler'));
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create domain profile');
      },
    }
  );

  const domainProfileActionMutation = useMutation(
    ({
      profileId,
      action,
    }: {
      profileId: string;
      action: 'start' | 'pause' | 'resume' | 'cancel' | 'run_now';
    }) => apiClient.performDomainResearchProfileAction(profileId, { action }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Domain profile action failed');
      },
    }
  );

  const updateDomainProfileMutation = useMutation(
    ({ profileId, data }: { profileId: string; data: DomainResearchProfileUpdate }) =>
      apiClient.updateDomainResearchProfile(profileId, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        toast.success('Domain profile settings updated');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update domain profile');
      },
    }
  );

  const createResearchPortfolioMutation = useMutation(
    (data: ResearchPortfolioCreate) => apiClient.createResearchPortfolio(data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Research fleet portfolio created');
        setPortfolioTitle('');
        setPortfolioObjective('');
        setPortfolioProfileSelection({});
        setPortfolioSandboxProfileId(resolveSandboxProfileId('generic'));
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create research portfolio');
      },
    }
  );

  const createScientificSandboxProfileMutation = useMutation(
    (data: ScientificSandboxProfileCreate) => apiClient.createScientificSandboxProfile(data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['scientific-sandbox-profiles']);
        toast.success('Scientific sandbox profile created');
        setEditingScientificSandboxProfileId('');
        setSandboxProfileDraft(buildScientificSandboxProfileDraft());
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create scientific sandbox profile');
      },
    }
  );

  const updateScientificSandboxProfileMutation = useMutation(
    ({ profileId, data }: { profileId: string; data: ScientificSandboxProfileUpdate }) =>
      apiClient.updateScientificSandboxProfile(profileId, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['scientific-sandbox-profiles']);
        toast.success('Scientific sandbox profile updated');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update scientific sandbox profile');
      },
    }
  );

  const deleteScientificSandboxProfileMutation = useMutation(
    (profileId: string) => apiClient.deleteScientificSandboxProfile(profileId),
    {
      onSuccess: (_, profileId) => {
        queryClient.invalidateQueries(['scientific-sandbox-profiles']);
        toast.success('Scientific sandbox profile deleted');
        if (String(editingScientificSandboxProfileId) === String(profileId)) {
          setEditingScientificSandboxProfileId('');
          setSandboxProfileDraft(buildScientificSandboxProfileDraft());
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to delete scientific sandbox profile');
      },
    }
  );
  const submitScientificSandboxDraft = useCallback(() => {
    const id = String(sandboxProfileDraft.id || '').trim();
    const name = String(sandboxProfileDraft.name || '').trim();
    if (!id || !name) {
      toast.error('Sandbox profile id and name are required');
      return;
    }
    const payload: ScientificSandboxProfileCreate = {
      id,
      name,
      description: String(sandboxProfileDraft.description || '').trim() || undefined,
      track_type: String(sandboxProfileDraft.track_type || 'generic').trim() || 'generic',
      backend: String(sandboxProfileDraft.backend || 'docker').trim() || 'docker',
      docker_image: String(sandboxProfileDraft.docker_image || '').trim() || undefined,
      timeout_seconds: Number(sandboxProfileDraft.timeout_seconds) > 0 ? Number(sandboxProfileDraft.timeout_seconds) : 900,
      resource_caps: {
        memory_mb: Number(sandboxProfileDraft.memory_mb) > 0 ? Number(sandboxProfileDraft.memory_mb) : 2048,
        cpus: Number(sandboxProfileDraft.cpus) > 0 ? Number(sandboxProfileDraft.cpus) : 1.5,
        pids_limit: Number(sandboxProfileDraft.pids_limit) > 0 ? Number(sandboxProfileDraft.pids_limit) : 192,
      },
      allowed_benchmark_families: splitUniqueLines(String(sandboxProfileDraft.allowed_benchmark_families || ''), 16),
      allowed_perf_collectors: splitUniqueLines(String(sandboxProfileDraft.allowed_perf_collectors || ''), 16),
      required_capabilities: splitUniqueLines(String(sandboxProfileDraft.required_capabilities || ''), 16),
      toolchains: splitUniqueLines(String(sandboxProfileDraft.toolchains || ''), 24),
      budget_limit_default: Number(sandboxProfileDraft.budget_limit_default) > 0 ? Number(sandboxProfileDraft.budget_limit_default) : 25,
      enabled: Boolean(sandboxProfileDraft.enabled),
      is_default: Boolean(sandboxProfileDraft.is_default),
    };
    if (editingScientificSandboxProfileId) {
      const updatePayload: ScientificSandboxProfileUpdate = editingScientificSandboxSystemManaged
        ? {
            name: payload.name,
            description: payload.description,
            enabled: payload.enabled,
            is_default: payload.is_default,
          }
        : payload;
      updateScientificSandboxProfileMutation.mutate({
        profileId: String(editingScientificSandboxProfileId),
        data: updatePayload,
      });
      return;
    }
    createScientificSandboxProfileMutation.mutate(payload);
  }, [
    createScientificSandboxProfileMutation,
    editingScientificSandboxProfileId,
    editingScientificSandboxSystemManaged,
    sandboxProfileDraft,
    updateScientificSandboxProfileMutation,
  ]);

  const createScientificResearchPackMutation = useMutation(
    async () => {
      const repoSourceIds = codeSources
        .map((source) => String((source as any)?.id || '').trim())
        .filter(Boolean);
      const blueprint = scientificResearchPackBlueprint(repoSourceIds);
      const compilerSandboxProfileId = resolveSandboxProfileId('compiler');
      const microarchitectureSandboxProfileId = resolveSandboxProfileId('microarchitecture');
      const compilerProfile = await apiClient.createDomainResearchProfile({
        ...blueprint.compiler,
        source_scope: blueprint.sourceScope,
        repo_source_ids: repoSourceIds,
        sandbox_profile_id: compilerSandboxProfileId,
        automation_profile: 'max_autonomy',
        automation_policy: {
          ...DEFAULT_VALIDATION_POLICY,
          auto_execute_validation_runs: true,
        },
        interval_minutes: 1440,
        persist_artifacts: true,
        auto_launch_follow_up: true,
        auto_create_experiment_plans: true,
        start_immediately: true,
      });
      const microarchitectureProfile = await apiClient.createDomainResearchProfile({
        ...blueprint.microarchitecture,
        source_scope: blueprint.sourceScope,
        repo_source_ids: repoSourceIds,
        sandbox_profile_id: microarchitectureSandboxProfileId,
        automation_profile: 'max_autonomy',
        automation_policy: {
          ...DEFAULT_VALIDATION_POLICY,
          auto_execute_validation_runs: true,
        },
        interval_minutes: 1440,
        persist_artifacts: true,
        auto_launch_follow_up: true,
        auto_create_experiment_plans: true,
        start_immediately: true,
      });
      return apiClient.createResearchPortfolio({
        title: blueprint.portfolio.title,
        objective: blueprint.portfolio.objective,
        linked_profile_ids: [compilerProfile.id, microarchitectureProfile.id],
        automation_profile: 'max_autonomy',
        automation_policy: {
          ...DEFAULT_VALIDATION_POLICY,
          auto_execute_validation_runs: true,
          auto_launch_experiment_runs: true,
          max_auto_follow_up_launches: 4,
          confidence_threshold: 0.68,
          experiment_readiness_threshold: 0.72,
          max_concurrent_validation_runs: 2,
          max_validation_runtime_minutes: 30,
          max_validation_budget_per_run: 50,
          duplicate_window_items: 120,
        },
        sandbox_profile_id: compilerSandboxProfileId,
        start_immediately: true,
      });
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Scientific research pack created');
        setActiveTab('fleet');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create scientific research pack');
      },
    }
  );

  const researchPortfolioActionMutation = useMutation(
    ({
      portfolioId,
      action,
    }: {
      portfolioId: string;
      action: 'start' | 'pause' | 'resume' | 'cancel' | 'run_now';
    }) => apiClient.performResearchPortfolioAction(portfolioId, { action }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Research portfolio action failed');
      },
    }
  );

  const updateResearchPortfolioMutation = useMutation(
    ({ portfolioId, data }: { portfolioId: string; data: ResearchPortfolioUpdate }) =>
      apiClient.updateResearchPortfolio(portfolioId, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-portfolios']);
        toast.success('Research fleet settings updated');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update research fleet');
      },
    }
  );

  const invalidateOpportunityExperimentQueries = useCallback(
    (ownerResponse: any, opportunityId?: string) => {
      const normalizedOpportunityId = String(opportunityId || '').trim();
      const opportunities = Array.isArray(ownerResponse?.opportunities) ? ownerResponse.opportunities : [];
      const matchedOpportunity = opportunities.find((row: any) => String(row?.opportunity_id || '').trim() === normalizedOpportunityId);
      const noteId = String((matchedOpportunity?.source_note_ids || [ownerResponse?.latest_note_ids?.[0] || ''])[0] || '').trim();
      const planId = String(
        matchedOpportunity?.latest_experiment_plan_id
        || (Array.isArray(matchedOpportunity?.linked_experiment_plan_ids) ? matchedOpportunity.linked_experiment_plan_ids[0] : '')
        || ''
      ).trim();
      if (noteId) {
        queryClient.invalidateQueries(['research-notes']);
        queryClient.invalidateQueries(['experiment-plans', noteId]);
      }
      if (planId) {
        queryClient.invalidateQueries(['experiment-runs', planId]);
      }
    },
    [queryClient]
  );

  const domainOpportunityActionMutation = useMutation(
    ({
      profileId,
      opportunityId,
      action,
      operatorNote,
      startImmediately,
    }: {
      profileId: string;
      opportunityId: string;
      action: 'accept' | 'suppress' | 'reopen' | 'create_plan' | 'launch_validation' | 'materialize_experiment' | 'launch_follow_up' | 'relaunch_follow_up';
      operatorNote?: string;
      startImmediately?: boolean;
    }) => apiClient.actOnDomainResearchOpportunity(profileId, opportunityId, {
      action,
      operator_note: operatorNote,
      start_immediately: startImmediately,
    }),
    {
      onSuccess: (response, variables) => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        if (variables?.action === 'materialize_experiment' || variables?.action === 'launch_validation') {
          invalidateOpportunityExperimentQueries(response, variables?.opportunityId);
        }
        if (variables?.action === 'launch_follow_up' || variables?.action === 'relaunch_follow_up') {
          queryClient.invalidateQueries(['research-inbox']);
          queryClient.invalidateQueries(['research-inbox-stats']);
          queryClient.invalidateQueries(['agent-checkpoint-queue']);
          queryClient.invalidateQueries(['agent-decision-trace']);
          queryClient.invalidateQueries(['agent-decision-trace-analytics']);
          queryClient.invalidateQueries(['notifications']);
          queryClient.invalidateQueries(['notifications-unread-count']);
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Opportunity action failed');
      },
    }
  );

  const researchPortfolioOpportunityActionMutation = useMutation(
    ({
      portfolioId,
      opportunityId,
      action,
      operatorNote,
      startImmediately,
    }: {
      portfolioId: string;
      opportunityId: string;
      action: 'accept' | 'suppress' | 'reopen' | 'create_plan' | 'launch_validation' | 'materialize_experiment' | 'launch_follow_up' | 'relaunch_follow_up';
      operatorNote?: string;
      startImmediately?: boolean;
    }) => apiClient.actOnResearchPortfolioOpportunity(portfolioId, opportunityId, {
      action,
      operator_note: operatorNote,
      start_immediately: startImmediately,
    }),
    {
      onSuccess: (response, variables) => {
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        if (variables?.action === 'materialize_experiment' || variables?.action === 'launch_validation') {
          invalidateOpportunityExperimentQueries(response, variables?.opportunityId);
        }
        if (variables?.action === 'launch_follow_up' || variables?.action === 'relaunch_follow_up') {
          queryClient.invalidateQueries(['research-inbox']);
          queryClient.invalidateQueries(['research-inbox-stats']);
          queryClient.invalidateQueries(['agent-checkpoint-queue']);
          queryClient.invalidateQueries(['agent-decision-trace']);
          queryClient.invalidateQueries(['agent-decision-trace-analytics']);
          queryClient.invalidateQueries(['notifications']);
          queryClient.invalidateQueries(['notifications-unread-count']);
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Opportunity action failed');
      },
    }
  );

  const updatePortfolioPolicyDraftField = useCallback(
    (portfolio: ResearchPortfolio, field: keyof ResearchPortfolioPolicyDraft, value: string | boolean) => {
      const portfolioId = String(portfolio.id || '');
      if (!portfolioId) return;
      setPortfolioPolicyDrafts((prev) => {
        const current = prev[portfolioId] || buildResearchPortfolioPolicyDraft(portfolio);
        return {
          ...prev,
          [portfolioId]: {
            ...current,
            [field]: value,
          },
        };
      });
    },
    []
  );

  const updateDomainProfilePolicyDraftField = useCallback(
    (profile: DomainResearchProfile, field: keyof DomainResearchProfilePolicyDraft, value: string | boolean) => {
      const profileId = String(profile.id || '');
      if (!profileId) return;
      setDomainProfilePolicyDrafts((prev) => {
        const current = prev[profileId] || buildDomainResearchProfilePolicyDraft(profile);
        return {
          ...prev,
          [profileId]: {
            ...current,
            [field]: value,
          },
        };
      });
    },
    []
  );

  const submitDomainProfilePolicyDraft = useCallback(
    (profile: DomainResearchProfile) => {
      const profileId = String(profile.id || '');
      const draft = domainProfilePolicyDrafts[profileId] || buildDomainResearchProfilePolicyDraft(profile);
      updateDomainProfileMutation.mutate({
        profileId,
        data: buildDomainResearchProfileUpdatePayload(draft),
      });
    },
    [domainProfilePolicyDrafts, updateDomainProfileMutation]
  );

  const submitPortfolioPolicyDraft = useCallback(
    (portfolio: ResearchPortfolio) => {
      const portfolioId = String(portfolio.id || '');
      const draft = portfolioPolicyDrafts[portfolioId] || buildResearchPortfolioPolicyDraft(portfolio);
      updateResearchPortfolioMutation.mutate({
        portfolioId,
        data: buildResearchPortfolioUpdatePayload(draft),
      });
    },
    [portfolioPolicyDrafts, updateResearchPortfolioMutation]
  );

  const beginOpportunityAction = useCallback(
    (mode: 'suppress' | 'launch' | 'relaunch', surface: 'domain' | 'fleet', ownerId: string, opportunity: ResearchOpportunity | Record<string, any>) => {
      setOpportunityNoteDraft({
        mode,
        surface,
        ownerId,
        opportunityId: String(opportunity.opportunity_id || ''),
        value: mode === 'suppress' ? String(opportunity.operator_note || '') : '',
      });
    },
    []
  );

  const beginOpportunitySuppression = useCallback(
    (surface: 'domain' | 'fleet', ownerId: string, opportunity: ResearchOpportunity) => {
      beginOpportunityAction('suppress', surface, ownerId, opportunity);
    },
    [beginOpportunityAction]
  );

  const beginOpportunityRelaunch = useCallback(
    (surface: 'domain' | 'fleet', ownerId: string, opportunity: ResearchOpportunity | Record<string, any>) => {
      beginOpportunityAction('relaunch', surface, ownerId, opportunity);
    },
    [beginOpportunityAction]
  );

  const beginOpportunityLaunch = useCallback(
    (surface: 'domain' | 'fleet', ownerId: string, opportunity: ResearchOpportunity | Record<string, any>) => {
      beginOpportunityAction('launch', surface, ownerId, opportunity);
    },
    [beginOpportunityAction]
  );

  const cancelOpportunityAction = useCallback(() => {
    setOpportunityNoteDraft(null);
  }, []);

  const submitOpportunityAction = useCallback(() => {
    if (!opportunityNoteDraft) return;
    const note = String(opportunityNoteDraft.value || '').trim();
    if (opportunityNoteDraft.mode === 'suppress' && !note) {
      toast.error('Suppression note is required');
      return;
    }
    const action = opportunityNoteDraft.mode === 'suppress'
      ? 'suppress'
      : opportunityNoteDraft.mode === 'launch'
        ? 'launch_follow_up'
        : 'relaunch_follow_up';
    if (opportunityNoteDraft.surface === 'domain') {
      domainOpportunityActionMutation.mutate({
        profileId: opportunityNoteDraft.ownerId,
        opportunityId: opportunityNoteDraft.opportunityId,
        action,
        operatorNote: note || undefined,
      });
    } else {
      researchPortfolioOpportunityActionMutation.mutate({
        portfolioId: opportunityNoteDraft.ownerId,
        opportunityId: opportunityNoteDraft.opportunityId,
        action,
        operatorNote: note || undefined,
      });
    }
    setOpportunityNoteDraft(null);
  }, [domainOpportunityActionMutation, opportunityNoteDraft, researchPortfolioOpportunityActionMutation]);

  const renderOpportunityExplainabilityPanel = useCallback((
    rowKey: string,
    row: Record<string, any>,
    context?: { surface: 'domain' | 'fleet'; ownerId: string } | null,
  ) => {
    const explanationRows = resolveOpportunityExplanationRows(row);
    const reprioritizationMeta = renderOpportunityReprioritizationMeta(row);
    const canRelaunch = Boolean(context?.ownerId && canRelaunchOpportunityRow(row));
    if (explanationRows.length === 0 && !reprioritizationMeta && !canRelaunch) return null;
    const isExpanded = Boolean(expandedOpportunityExplanationRows[rowKey]) || highlightedAutonomyRowKey === rowKey;
    const isRelaunchDraft = opportunityNoteDraft?.mode === 'relaunch'
      && opportunityNoteDraft.surface === context?.surface
      && String(opportunityNoteDraft.ownerId) === String(context?.ownerId || '')
      && String(opportunityNoteDraft.opportunityId) === String(row.opportunity_id || '');
    return (
      <div className="mt-2 rounded border border-slate-200 bg-slate-50 p-2 text-[11px] text-slate-700">
        <div className="flex items-center justify-between gap-2">
          <div className="font-medium text-slate-800">{resolveOpportunityExplanationHeading(row)}</div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setExpandedOpportunityExplanationRows((prev) => ({ ...prev, [rowKey]: !isExpanded }))}
          >
            {isExpanded ? 'Hide details' : 'Show details'}
          </Button>
        </div>
        {isExpanded ? (
          <div className="mt-2 space-y-1">
            {explanationRows.map((item) => (
              <div key={`${rowKey}-${item.label}`}>
                <span className="font-medium text-slate-800">{item.label}:</span>{' '}
                <span>{item.value}</span>
              </div>
            ))}
            {reprioritizationMeta}
            {canRelaunch ? (
              <div className="pt-1">
                {isRelaunchDraft ? (
                  <div className="rounded border border-emerald-200 bg-emerald-50 p-2">
                    <div className="text-[11px] font-medium text-emerald-700">Relaunch note</div>
                    <textarea
                      aria-label={`${context?.surface === 'fleet' ? 'Fleet' : 'Domain'} relaunch note`}
                      className="mt-2 w-full border border-emerald-200 rounded px-2 py-1 text-xs"
                      rows={3}
                      value={opportunityNoteDraft?.value || ''}
                      onChange={(e) => setOpportunityNoteDraft((prev) => prev ? { ...prev, value: e.target.value } : prev)}
                    />
                    <div className="mt-2 flex gap-2">
                      <Button size="sm" variant="secondary" onClick={submitOpportunityAction}>
                        Relaunch follow-up
                      </Button>
                      <Button size="sm" variant="ghost" onClick={cancelOpportunityAction}>
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => {
                      if (!context?.ownerId || !context.surface) return;
                      beginOpportunityRelaunch(context.surface, context.ownerId, row);
                    }}
                  >
                    Relaunch Follow-up
                  </Button>
                )}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    );
  }, [beginOpportunityRelaunch, cancelOpportunityAction, expandedOpportunityExplanationRows, highlightedAutonomyRowKey, opportunityNoteDraft, submitOpportunityAction]);

  const renderManualRecommendationAction = useCallback((
    surface: 'domain' | 'fleet',
    ownerId: string,
    row: Record<string, any>,
  ) => {
    const opportunityId = String(row.opportunity_id || '').trim();
    if (!ownerId || !opportunityId) return null;
    const isRelaunch = canRelaunchOpportunityRow(row);
    const hasChildJobs = Array.isArray(row.child_job_ids) && row.child_job_ids.length > 0;
    if (!isRelaunch && hasChildJobs) return null;
    const isDraftOpen = opportunityNoteDraft?.surface === surface
      && String(opportunityNoteDraft.ownerId) === String(ownerId)
      && String(opportunityNoteDraft.opportunityId) === opportunityId
      && (opportunityNoteDraft.mode === 'launch' || opportunityNoteDraft.mode === 'relaunch');
    const labelPrefix = surface === 'fleet' ? 'Fleet' : 'Domain';
    return (
      <div className="mt-2">
        {isDraftOpen ? (
          <div className="rounded border border-emerald-200 bg-emerald-50 p-2">
            <div className="text-[11px] font-medium text-emerald-700">
              {isRelaunch ? 'Relaunch note' : 'Follow-up note'}
            </div>
            <textarea
              aria-label={`${labelPrefix} ${isRelaunch ? 'relaunch' : 'follow-up'} note`}
              className="mt-2 w-full border border-emerald-200 rounded px-2 py-1 text-xs"
              rows={3}
              value={opportunityNoteDraft?.value || ''}
              onChange={(e) => setOpportunityNoteDraft((prev) => prev ? { ...prev, value: e.target.value } : prev)}
            />
            <div className="mt-2 flex gap-2">
              <Button size="sm" variant="secondary" onClick={submitOpportunityAction}>
                {isRelaunch ? 'Relaunch follow-up' : 'Launch follow-up'}
              </Button>
              <Button size="sm" variant="ghost" onClick={cancelOpportunityAction}>
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              if (isRelaunch) {
                beginOpportunityRelaunch(surface, ownerId, row);
                return;
              }
              beginOpportunityLaunch(surface, ownerId, row);
            }}
          >
            {isRelaunch ? 'Relaunch Follow-up' : 'Follow-up'}
          </Button>
        )}
      </div>
    );
  }, [beginOpportunityLaunch, beginOpportunityRelaunch, cancelOpportunityAction, opportunityNoteDraft, submitOpportunityAction]);

  const codingBacklogActionMutation = useMutation(
    ({
      itemId,
      action,
      sliceId,
      assignedUserId,
      closureReason,
      operatorNote,
    }: {
      itemId: string;
      action:
        | 'start'
        | 'pause'
        | 'resume'
        | 'cancel'
        | 'close'
        | 'assign_backlog'
        | 'clear_backlog_assignment'
        | 'update_backlog_note'
        | 'apply_override'
        | 'create_patch_pr'
        | 'keep_proposal_only'
        | 'relaunch_slice'
        | 'skip_slice';
      sliceId?: string;
      assignedUserId?: string;
      closureReason?: string;
      operatorNote?: string;
    }) =>
      apiClient.performCodingBacklogAction(itemId, {
        action,
        slice_id: sliceId,
        assigned_user_id: assignedUserId,
        closure_reason: closureReason,
        operator_note: operatorNote,
      }),
    {
      onSuccess: (_, vars) => {
        queryClient.invalidateQueries(['coding-backlog-items']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        if (vars?.action === 'assign_backlog') {
          toast.success('Backlog assignment updated');
        } else if (vars?.action === 'clear_backlog_assignment') {
          toast.success('Backlog assignment cleared');
        } else if (vars?.action === 'update_backlog_note') {
          setBacklogNoteDrafts((prev) => {
            const next = { ...prev };
            delete next[String(vars.itemId || '')];
            return next;
          });
          toast.success('Backlog note saved');
        } else if (vars?.action === 'close' || vars?.action === 'cancel') {
          setBacklogCloseReasonDrafts((prev) => {
            const next = { ...prev };
            delete next[String(vars.itemId || '')];
            return next;
          });
          setBacklogNoteDrafts((prev) => {
            const next = { ...prev };
            delete next[String(vars.itemId || '')];
            return next;
          });
          toast.success('Backlog item closed');
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Coding backlog action failed');
      },
    }
  );

  const createInboxMonitorMutation = useMutation(
    (data: AgentJobCreate) => apiClient.createAgentJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        toast.success('Monitor created');
        setShowInboxMonitorModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Create failed');
      },
    }
  );

  const createFromTemplateMutation = useMutation(
    (data: AgentJobFromTemplate) => apiClient.createAgentJobFromTemplate(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        toast.success('Job created from template');
        setCreateFromTemplate(null);
        setShowCustomerResearchModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Create failed');
      },
    }
  );

  const createFromChainMutation = useMutation(
    (data: AgentJobFromChainCreate) => apiClient.createJobFromChain(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        toast.success('Chain started');
        setStartFromChain(null);
        setShowCustomerResearchModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to start chain');
      },
    }
  );

  const launchQueueRecommendation = useCallback(
    (item: AgentCheckpointQueueItem, payload: Record<string, any>) => {
      if (payload?.chain_definition_id) {
        createFromChainMutation.mutate(payload as AgentJobFromChainCreate);
        return;
      }
      if (payload?.job_type && payload?.goal) {
        createMutation.mutate(payload as AgentJobCreate);
        return;
      }
      toast.error(`Queue item ${item.title} is missing a launch payload`);
    },
    [createFromChainMutation, createMutation]
  );

  const queueCustomerOptions = useMemo(
    () => Object.entries(checkpointQueueData?.by_customer || {}).filter(([customer]) => String(customer || '').trim()),
    [checkpointQueueData]
  );
  const isCompilerQueueItem = useCallback((item: AgentCheckpointQueueItem) => (
    String(item.track_type || '').trim().toLowerCase() === 'compiler'
  ), []);
  const visibleQueueItems = useMemo(
    () => ((checkpointQueueData?.items || []) as AgentCheckpointQueueItem[]).filter((item) => {
      if (queueJobFilter && String(item.job_id || '').trim() !== queueJobFilter) return false;
      if (queueOperatorPreset === 'compiler' && !isCompilerQueueItem(item)) return false;
      if (queueOperatorPreset === 'approval_required') {
        if (!isCompilerQueueItem(item)) return false;
        if (String(item.item_type || '').trim() !== 'follow_up_recommendation') return false;
        const launchStatus = String(item.follow_up_launch_status || '').trim().toLowerCase();
        if (launchStatus !== 'pending_approval' && String(item.status || '').trim().toLowerCase() !== 'pending_approval') return false;
      }
      if (queueOperatorPreset === 'blocked_validation') {
        if (!isCompilerQueueItem(item)) return false;
        if (!['policy_review', 'budget_review'].includes(String(item.item_type || '').trim())) return false;
      }
      if (queueOperatorPreset === 'failed_follow_up') {
        const schedulerStatus = String(item.scheduler_state?.last_run_status || '').trim().toLowerCase();
        if (!isCompilerQueueItem(item)) return false;
        if (!['failed', 'cancelled'].includes(schedulerStatus)) return false;
        if (!((item.child_job_ids || []).length || String(item.follow_up_job_id || '').trim())) return false;
      }
      if (!queueHealthDrilldown) return true;
      if (String(item.item_type || '').trim() !== 'follow_up_recommendation') return false;
      const launchStatus = String(item.follow_up_launch_status || '').trim().toLowerCase();
      const followUpDecision = String(item.follow_up_decision || '').trim().toLowerCase();
      const reasonCode = String(item.reason_code || '').trim().toLowerCase();
      if (queueHealthDrilldown === 'pending_follow_up_approvals') {
        return launchStatus === 'pending_approval' || String(item.status || '').trim().toLowerCase() === 'pending_approval';
      }
      if (queueHealthDrilldown === 'manual_follow_up_recommendations') {
        return launchStatus === 'blocked' || ['manual', 'manual_only', 'manual_recommendation'].includes(followUpDecision);
      }
      if (queueHealthDrilldown === 'blocked_follow_up') {
        return launchStatus === 'blocked' || reasonCode === 'follow_up_blocked';
      }
      return true;
    }),
    [checkpointQueueData, isCompilerQueueItem, queueHealthDrilldown, queueJobFilter, queueOperatorPreset]
  );
  const selectedQueueItems = useMemo(
    () => visibleQueueItems.filter((item) => queueSelection[item.queue_key]),
    [queueSelection, visibleQueueItems]
  );
  const toggleQueueSelection = useCallback((item: AgentCheckpointQueueItem) => {
    setQueueSelection((prev) => {
      const next = { ...prev };
      if (next[item.queue_key]) delete next[item.queue_key];
      else next[item.queue_key] = true;
      return next;
    });
  }, []);

  const selectVisibleQueueItems = useCallback(() => {
    setQueueSelection((prev) => {
      const next = { ...prev };
      visibleQueueItems.forEach((item) => {
        next[item.queue_key] = true;
      });
      return next;
    });
  }, [visibleQueueItems]);

  const clearQueueSelection = useCallback(() => {
    setQueueSelection({});
  }, []);

  const visibleInboxItems = useMemo(
    () => ((inboxData?.items || []) as ResearchInboxItem[]).filter((item) => {
      if (!inboxHealthDrilldown) return true;
      if (String(item.status || '').trim().toLowerCase() !== 'accepted') return false;
      if (String(item.item_type || '').trim() !== 'follow_up_recommendation') return false;
      const outcomeStatus = String(item.follow_up_outcome_status || '').trim().toLowerCase();
      const operatorDecision = String(item.follow_up_operator_decision || '').trim().toLowerCase();
      if (inboxHealthDrilldown === 'completed_follow_up') return outcomeStatus === 'completed';
      if (inboxHealthDrilldown === 'failed_follow_up') return outcomeStatus === 'failed';
      if (inboxHealthDrilldown === 'cancelled_follow_up') return outcomeStatus === 'cancelled';
      if (inboxHealthDrilldown === 'suppressed_relaunch') return operatorDecision === 'rejected';
      return true;
    }),
    [inboxData, inboxHealthDrilldown]
  );

  const selectedInboxItems = useMemo(
    () => visibleInboxItems.filter((item) => selectedInboxIds[item.id]),
    [selectedInboxIds, visibleInboxItems]
  );

  useEffect(() => {
    const visibleIds = new Set(visibleInboxItems.map((item) => String(item.id)));
    setSelectedInboxIds((prev) => {
      const nextEntries = Object.entries(prev).filter(([id, enabled]) => enabled && visibleIds.has(id));
      if (nextEntries.length === Object.keys(prev).length) return prev;
      return Object.fromEntries(nextEntries);
    });
  }, [visibleInboxItems]);

  useEffect(() => {
    const visibleKeys = new Set(visibleQueueItems.map((item) => item.queue_key));
    setQueueSelection((prev) => {
      const nextEntries = Object.entries(prev).filter(([key, enabled]) => enabled && visibleKeys.has(key));
      if (nextEntries.length === Object.keys(prev).length) return prev;
      return Object.fromEntries(nextEntries);
    });
  }, [visibleQueueItems]);

  const getQueueDraft = useCallback((item: AgentCheckpointQueueItem) => {
    const action = (item.checkpoint?.action && typeof item.checkpoint.action === 'object')
      ? item.checkpoint.action as Record<string, any>
      : {};
    const params = (action.params && typeof action.params === 'object') ? action.params : {};
    return {
      note: '',
      showEdit: false,
      tool: String(action.tool || ''),
      purpose: String(action.purpose || ''),
      params: JSON.stringify(params, null, 2),
    };
  }, []);

  const getQueueDraftValue = useCallback((item: AgentCheckpointQueueItem) => {
    return queueDrafts[item.queue_key] || getQueueDraft(item);
  }, [getQueueDraft, queueDrafts]);

  const queueBulkState = useMemo(() => {
    if (selectedQueueItems.length === 0) {
      return {
        itemType: null as 'approval_checkpoint' | 'job_recovery' | 'follow_up_recommendation' | null,
        actions: [] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
        disabledReason: 'Select one or more queue items to use bulk triage.',
      };
    }

    const itemTypes = Array.from(new Set(selectedQueueItems.map((item) => String(item.item_type || '').trim())));
    if (itemTypes.length !== 1) {
      return {
        itemType: null as 'approval_checkpoint' | 'job_recovery' | 'follow_up_recommendation' | null,
        actions: [] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
        disabledReason: 'Bulk actions only support selections with one queue item type.',
      };
    }

    const itemType = itemTypes[0];
    if (itemType === 'follow_up_recommendation') {
      const followUpRows = selectedQueueItems.map((item) => {
        const profileId = String(item.domain_research_profile_id || '').trim();
        const profileOpportunityId = String(item.profile_opportunity_id || '').trim();
        const portfolioId = String(item.portfolio_id || '').trim();
        const portfolioOpportunityId = String(item.portfolio_opportunity_id || '').trim();
        const ownerKind = profileId && profileOpportunityId
          ? 'domain'
          : portfolioId && portfolioOpportunityId
            ? 'fleet'
            : '';
        const ownerId = ownerKind === 'domain' ? profileId : ownerKind === 'fleet' ? portfolioId : '';
        const opportunityId = ownerKind === 'domain' ? profileOpportunityId : ownerKind === 'fleet' ? portfolioOpportunityId : '';
        return {
          ownerKind,
          ownerId,
          opportunityId,
          pendingApproval: String(item.follow_up_launch_status || '').trim().toLowerCase() === 'pending_approval',
        };
      });
      if (followUpRows.some((row) => !row.ownerKind || !row.ownerId || !row.opportunityId)) {
        return {
          itemType: null as 'approval_checkpoint' | 'job_recovery' | 'follow_up_recommendation' | null,
          actions: [] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
          disabledReason: 'Selected follow-up items are missing owner or opportunity identifiers.',
        };
      }
      if (followUpRows.some((row) => !row.pendingApproval)) {
        return {
          itemType: null as 'approval_checkpoint' | 'job_recovery' | 'follow_up_recommendation' | null,
          actions: [] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
          disabledReason: 'Bulk follow-up actions only support pending approvals.',
        };
      }
      const ownerKinds = Array.from(new Set(followUpRows.map((row) => row.ownerKind)));
      if (ownerKinds.length !== 1) {
        return {
          itemType: null as 'approval_checkpoint' | 'job_recovery' | 'follow_up_recommendation' | null,
          actions: [] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
          disabledReason: 'Bulk follow-up actions cannot mix domain and fleet owners.',
        };
      }
      const ownerIds = Array.from(new Set(followUpRows.map((row) => row.ownerId)));
      if (ownerIds.length !== 1) {
        return {
          itemType: null as 'approval_checkpoint' | 'job_recovery' | 'follow_up_recommendation' | null,
          actions: [] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
          disabledReason: 'Bulk follow-up actions must stay within one domain profile or research fleet.',
        };
      }
      return {
        itemType: 'follow_up_recommendation' as const,
        actions: ['approve_launch', 'reject_launch'] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
        disabledReason: '',
      };
    }
    if (itemType !== 'approval_checkpoint' && itemType !== 'job_recovery') {
      return {
        itemType: null as 'approval_checkpoint' | 'job_recovery' | 'follow_up_recommendation' | null,
        actions: [] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
        disabledReason: 'Selected items do not support bulk actions.',
      };
    }
    if (selectedQueueItems.some((item) => !item.job_id)) {
      return {
        itemType: null as 'approval_checkpoint' | 'job_recovery' | 'follow_up_recommendation' | null,
        actions: [] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
        disabledReason: 'Bulk actions only support queue items backed by jobs.',
      };
    }
    if (itemType === 'approval_checkpoint') {
      const hasInlineEdit = selectedQueueItems.some((item) => getQueueDraftValue(item).showEdit);
      if (hasInlineEdit) {
        return {
          itemType: 'approval_checkpoint' as const,
          actions: [] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
          disabledReason: 'Bulk approval actions are disabled while any selected item is in Edit Action mode.',
        };
      }
      return {
        itemType: 'approval_checkpoint' as const,
        actions: ['approve', 'skip', 'reject'] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
        disabledReason: '',
      };
    }
    return {
      itemType: 'job_recovery' as const,
      actions: ['restart', 'resume', 'cancel'] as Array<'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | 'approve_launch' | 'reject_launch'>,
      disabledReason: '',
    };
  }, [getQueueDraftValue, selectedQueueItems]);

  const inboxBulkFollowUpState = useMemo(() => {
    if (selectedInboxItems.length === 0) {
      return {
        enabled: false,
        disabledReason: 'Select one or more inbox items to use bulk follow-up actions.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    if (selectedInboxItems.some((item) => String(item.item_type || '').trim() !== 'follow_up_recommendation')) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk follow-up actions only support follow-up recommendations.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    const followUpRows = selectedInboxItems.map((item) => {
      const sourceKind = String(item.origin_source_kind || '').trim().toLowerCase();
      const ownerKind = sourceKind === 'profile'
        ? 'domain'
        : sourceKind === 'portfolio'
          ? 'fleet'
          : '';
      return {
        ownerKind,
        ownerId: String(item.origin_source_id || '').trim(),
        opportunityId: String(item.origin_opportunity_id || '').trim(),
        pendingApproval: String(item.follow_up_launch_status || '').trim().toLowerCase() === 'pending_approval',
      };
    });
    if (followUpRows.some((row) => !row.pendingApproval)) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk follow-up actions only support pending approvals.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    if (followUpRows.some((row) => !row.ownerKind || !row.ownerId || !row.opportunityId)) {
      return {
        enabled: false,
        disabledReason: 'Selected inbox follow-up items are missing owner or opportunity identifiers.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    const ownerKinds = Array.from(new Set(followUpRows.map((row) => row.ownerKind)));
    if (ownerKinds.length !== 1) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk follow-up actions cannot mix domain and fleet owners.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    const ownerIds = Array.from(new Set(followUpRows.map((row) => row.ownerId)));
    if (ownerIds.length !== 1) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk follow-up actions must stay within one domain profile or research fleet.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    return {
      enabled: true,
      disabledReason: '',
      ownerKind: ownerKinds[0] as 'domain' | 'fleet',
      ownerId: ownerIds[0],
      opportunityIds: Array.from(new Set(followUpRows.map((row) => row.opportunityId))),
    };
  }, [selectedInboxItems]);

  const inboxBulkRelaunchState = useMemo(() => {
    if (selectedInboxItems.length === 0) {
      return {
        enabled: false,
        disabledReason: 'Select one or more inbox items to use bulk relaunch.',
        itemIds: [] as string[],
      };
    }
    if (selectedInboxItems.some((item) => String(item.item_type || '').trim() !== 'follow_up_recommendation')) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk relaunch only supports follow-up recommendations.',
        itemIds: [] as string[],
      };
    }
    if (
      selectedInboxItems.some(
        (item) =>
          String(item.follow_up_launch_status || '').trim().toLowerCase() !== 'launched'
          || !['failed', 'cancelled'].includes(String(item.follow_up_outcome_status || '').trim().toLowerCase())
      )
    ) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk relaunch only supports failed or cancelled launched follow-ups.',
        itemIds: [] as string[],
      };
    }
    return {
      enabled: true,
      disabledReason: '',
      itemIds: selectedInboxItems.map((item) => String(item.id)),
    };
  }, [selectedInboxItems]);

  const setQueueDraftValue = useCallback((item: AgentCheckpointQueueItem, patch: Partial<{
    note: string;
    showEdit: boolean;
    tool: string;
    purpose: string;
    params: string;
  }>) => {
    setQueueDrafts((prev) => ({
      ...prev,
      [item.queue_key]: {
        ...(prev[item.queue_key] || getQueueDraft(item)),
        ...patch,
      },
    }));
  }, [getQueueDraft]);

  const openQueueItemTarget = useCallback((item: AgentCheckpointQueueItem) => {
    if (item.domain_research_profile_id) {
      setActiveTab('domain');
      navigate(buildAutonomousAgentsUrl(undefined, {
        tab: 'domain',
        profileId: String(item.domain_research_profile_id),
        opportunityId: item.profile_opportunity_id ? String(item.profile_opportunity_id) : undefined,
      }), { replace: true });
      return;
    }
    if (item.portfolio_id) {
      setActiveTab('fleet');
      navigate(buildAutonomousAgentsUrl(undefined, {
        tab: 'fleet',
        fleetId: String(item.portfolio_id),
        opportunityId: item.portfolio_opportunity_id ? String(item.portfolio_opportunity_id) : undefined,
      }), { replace: true });
      return;
    }
    if (item.job_id) {
      if (item.job) {
        setSelectedJob(item.job);
      }
      setActiveTab('jobs');
      navigate(buildAutonomousAgentsUrl(String(item.job_id)), { replace: true });
    }
  }, [buildAutonomousAgentsUrl, navigate]);

  const runQueueAction = useCallback((item: AgentCheckpointQueueItem, action: 'approve' | 'reject' | 'skip' | 'edit' | 'restart' | 'resume' | 'cancel') => {
    if (!item.job_id) return;
    const draft = getQueueDraftValue(item);
    if (action === 'edit') {
      let parsedParams: Record<string, any> = {};
      try {
        parsedParams = draft.params.trim() ? JSON.parse(draft.params) : {};
      } catch (error) {
        toast.error('Approval params must be valid JSON');
        return;
      }
      const patch: Record<string, any> = {};
      if (draft.tool.trim()) patch.tool = draft.tool.trim();
      if (draft.purpose.trim()) patch.purpose = draft.purpose.trim();
      patch.params = parsedParams;
      actionMutation.mutate({
        jobId: String(item.job_id),
        action,
        checkpointNote: draft.note.trim() || undefined,
        checkpointActionPatch: patch,
      });
      return;
    }
    actionMutation.mutate({
      jobId: String(item.job_id),
      action,
      checkpointNote: draft.note.trim() || undefined,
    });
  }, [actionMutation, getQueueDraftValue]);

  const runQueuePolicyAction = useCallback((item: AgentCheckpointQueueItem, action: AgentCheckpointQueueAction) => {
    if (action.action === 'open_fleet') {
      openQueueItemTarget(item);
      return;
    }
    const monitorJobId = String(item.job_id || '').trim();
    const rollbackPayload = action.policy_rollback_payload || {};
    const updatePayload = action.policy_update_payload || {};
    if (!monitorJobId) {
      toast.error('Missing monitor id for policy action');
      return;
    }
    if (action.action === 'compare_before_after') {
      openHealthPolicyComparison(monitorJobId, String(item.policy_guardrail_target_history_entry_id || rollbackPayload.history_entry_id || '').trim() || undefined);
      return;
    }
    if (action.action === 'open_monitor') {
      setActiveTab('health');
      if (item.customer) {
        setHealthCustomerFilter(String(item.customer));
      }
      return;
    }
    if (action.action === 'apply_guardrail') {
      if (rollbackPayload.history_entry_id) {
        rollbackMonitorPolicyMutation.mutate({
          monitorJobId,
          historyEntryId: String(rollbackPayload.history_entry_id),
        });
        return;
      }
      updateMonitorPolicyMutation.mutate({
        monitorJobId,
        data: {
          ...updatePayload,
          change_source: 'policy_guardrail',
          change_reason: 'Applied from queue policy safeguard review',
        },
      });
    }
  }, [openHealthPolicyComparison, openQueueItemTarget, rollbackMonitorPolicyMutation, updateMonitorPolicyMutation]);

  const followUpQueueActionMutation = useMutation(
    ({
      inbox_item_id,
      domain_research_profile_id,
      profile_opportunity_id,
      portfolio_id,
      portfolio_opportunity_id,
      action,
      operator_note,
      navigateOnLaunch,
      refreshTarget,
      reviewRowKey,
    }: {
      inbox_item_id?: string;
      domain_research_profile_id?: string;
      profile_opportunity_id?: string;
      portfolio_id?: string;
      portfolio_opportunity_id?: string;
      action: 'approve_launch' | 'reject_launch';
      operator_note?: string;
      navigateOnLaunch?: boolean;
      refreshTarget?: 'domain' | 'fleet';
      reviewRowKey?: string;
    }) => apiClient.actionAgentCheckpointQueueFollowUp({
      inbox_item_id,
      domain_research_profile_id,
      profile_opportunity_id,
      portfolio_id,
      portfolio_opportunity_id,
      action,
      operator_note,
    }),
    {
      onMutate: (variables) => {
        if (variables.reviewRowKey) {
          setActiveFollowUpReviewKey(variables.reviewRowKey);
        }
      },
      onSuccess: (response, variables) => {
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        if (variables.refreshTarget === 'domain') {
          void refetchDomainProfiles();
        } else if (variables.refreshTarget === 'fleet') {
          void refetchResearchPortfolios();
        }
        if (variables.reviewRowKey) {
          const reviewKey = String(variables.reviewRowKey);
          setFollowUpReviewNoteDrafts((prev) => {
            if (!(reviewKey in prev)) return prev;
            const next = { ...prev };
            delete next[reviewKey];
            return next;
          });
        }
        if (response.follow_up_job_id) {
          toast.success('Follow-up launched');
          if (variables.navigateOnLaunch !== false) {
            setActiveTab('jobs');
            navigate(buildAutonomousAgentsUrl(String(response.follow_up_job_id)), { replace: true });
            return;
          }
        } else {
          toast.success(response.detail || 'Follow-up decision recorded');
        }
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to apply follow-up queue action');
      },
      onSettled: (_data, _error, variables) => {
        if (variables?.reviewRowKey) {
          setActiveFollowUpReviewKey((current) => (current === variables.reviewRowKey ? '' : current));
        }
      },
    }
  );

  const bulkFollowUpQueueActionMutation = useMutation(
    ({
      domain_research_profile_id,
      profile_opportunity_ids,
      portfolio_id,
      portfolio_opportunity_ids,
      action,
      operator_note,
      ownerKey,
      refreshTarget,
    }: {
      domain_research_profile_id?: string;
      profile_opportunity_ids?: string[];
      portfolio_id?: string;
      portfolio_opportunity_ids?: string[];
      action: 'approve_launch' | 'reject_launch';
      operator_note?: string;
      ownerKey: string;
      refreshTarget: 'domain' | 'fleet';
    }) => apiClient.bulkActionAgentCheckpointQueueFollowUp({
      domain_research_profile_id,
      profile_opportunity_ids,
      portfolio_id,
      portfolio_opportunity_ids,
      action,
      operator_note,
    }),
    {
      onMutate: (variables) => {
        setActiveBulkFollowUpOwnerKey(String(variables.ownerKey || ''));
      },
      onSuccess: (response, variables) => {
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        if (variables.refreshTarget === 'domain') {
          void refetchDomainProfiles();
        } else {
          void refetchResearchPortfolios();
        }
        const successfulIds = new Set(
          response.results
            .filter((row) => row.ok)
            .map((row) => String(row.profile_opportunity_id || row.portfolio_opportunity_id || '').trim())
            .filter(Boolean)
        );
        if (successfulIds.size > 0) {
          setBulkFollowUpSelection((prev) => {
            const next = { ...prev };
            successfulIds.forEach((opportunityId) => {
              delete next[`${variables.ownerKey}:${opportunityId}`];
            });
            return next;
          });
        }
        if (response.failed === 0) {
          setBulkFollowUpNotes((prev) => {
            if (!(variables.ownerKey in prev)) return prev;
            const next = { ...prev };
            delete next[variables.ownerKey];
            return next;
          });
          toast.success(
            variables.action === 'approve_launch'
              ? `Launched ${response.applied} follow-up${response.applied === 1 ? '' : 's'}`
              : `Rejected ${response.applied} follow-up${response.applied === 1 ? '' : 's'}`
          );
          return;
        }
        const failedLabels = response.results
          .filter((row) => !row.ok)
          .slice(0, 3)
          .map((row) => `${String(row.profile_opportunity_id || row.portfolio_opportunity_id || '').slice(0, 20)}: ${row.error || 'failed'}`);
        toast.error(`Applied ${response.applied}/${response.requested_count}. ${failedLabels.join(' | ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Bulk follow-up action failed');
      },
      onSettled: () => {
        setActiveBulkFollowUpOwnerKey('');
      },
    }
  );

  const bulkManualFollowUpActionMutation = useMutation(
    async ({
      scope,
      ownerId,
      opportunityIds,
      action,
      operator_note,
    }: {
      scope: 'domain' | 'fleet';
      ownerId: string;
      opportunityIds: string[];
      action: 'launch_follow_up' | 'relaunch_follow_up';
      operator_note?: string;
    }) => {
      const results = await Promise.all(
        opportunityIds.map(async (opportunityId) => {
          try {
            if (scope === 'domain') {
              await apiClient.actOnDomainResearchOpportunity(ownerId, opportunityId, {
                action,
                operator_note,
              });
            } else {
              await apiClient.actOnResearchPortfolioOpportunity(ownerId, opportunityId, {
                action,
                operator_note,
              });
            }
            return {
              opportunity_id: opportunityId,
              ok: true,
              action,
            };
          } catch (error: any) {
            return {
              opportunity_id: opportunityId,
              ok: false,
              action,
              error: error?.response?.data?.detail || error?.message || 'failed',
            };
          }
        })
      );
      return {
        requested_count: opportunityIds.length,
        applied: results.filter((row) => row.ok).length,
        failed: results.filter((row) => !row.ok).length,
        results,
      };
    },
    {
      onMutate: (variables) => {
        setActiveBulkFollowUpOwnerKey(buildBulkFollowUpOwnerKey(variables.scope, variables.ownerId));
      },
      onSuccess: (response, variables) => {
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['agent-decision-trace']);
        queryClient.invalidateQueries(['agent-decision-trace-analytics']);
        queryClient.invalidateQueries(['notifications']);
        queryClient.invalidateQueries(['notifications-unread-count']);
        if (variables.scope === 'domain') {
          void refetchDomainProfiles();
        } else {
          void refetchResearchPortfolios();
        }
        const ownerKey = buildBulkFollowUpOwnerKey(variables.scope, variables.ownerId);
        const successfulIds = new Set(
          response.results
            .filter((row) => row.ok)
            .map((row) => String(row.opportunity_id || '').trim())
            .filter(Boolean)
        );
        if (successfulIds.size > 0) {
          setBulkFollowUpSelection((prev) => {
            const next = { ...prev };
            successfulIds.forEach((opportunityId) => {
              delete next[`${ownerKey}:${opportunityId}`];
            });
            return next;
          });
        }
        if (response.failed === 0) {
          setBulkFollowUpNotes((prev) => {
            if (!(ownerKey in prev)) return prev;
            const next = { ...prev };
            delete next[ownerKey];
            return next;
          });
          toast.success(
            `${variables.action === 'relaunch_follow_up' ? 'Relaunched' : 'Launched'} ${response.applied} follow-up${response.applied === 1 ? '' : 's'}`
          );
          return;
        }
        const failedLabels = response.results
          .filter((row) => !row.ok)
          .slice(0, 3)
          .map((row) => `${String(row.opportunity_id || '').slice(0, 20)}: ${row.error || 'failed'}`);
        toast.error(`Applied ${response.applied}/${response.requested_count}. ${failedLabels.join(' | ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Bulk manual follow-up action failed');
      },
      onSettled: () => {
        setActiveBulkFollowUpOwnerKey('');
      },
    }
  );

  const bulkQueueActionMutation = useMutation(
    ({
      itemType,
      action,
      jobIds,
      checkpointNote,
    }: {
      itemType: 'approval_checkpoint' | 'job_recovery';
      action: 'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel';
      jobIds: string[];
      checkpointNote?: string;
    }) => apiClient.bulkActionAgentCheckpointQueue({
      item_type: itemType,
      action,
      job_ids: jobIds,
      checkpoint_note: checkpointNote,
    }),
    {
      onSuccess: (response) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        setQueueSelection({});
        setQueueBulkNote('');
        if (response.failed > 0) {
          const failedLabels = response.results
            .filter((row) => !row.ok)
            .slice(0, 3)
            .map((row) => `${String(row.job_id).slice(0, 8)}: ${row.error || 'failed'}`);
          toast.error(`Applied ${response.applied}/${response.requested_count}. ${failedLabels.join(' | ')}`);
          return;
        }
        toast.success(`Applied ${response.applied} queue action${response.applied === 1 ? '' : 's'}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Bulk queue action failed');
      },
    }
  );

  const bulkQueueFollowUpActionMutation = useMutation(
    ({
      domain_research_profile_id,
      profile_opportunity_ids,
      portfolio_id,
      portfolio_opportunity_ids,
      action,
      operator_note,
    }: {
      domain_research_profile_id?: string;
      profile_opportunity_ids?: string[];
      portfolio_id?: string;
      portfolio_opportunity_ids?: string[];
      action: 'approve_launch' | 'reject_launch';
      operator_note?: string;
    }) => apiClient.bulkActionAgentCheckpointQueueFollowUp({
      domain_research_profile_id,
      profile_opportunity_ids,
      portfolio_id,
      portfolio_opportunity_ids,
      action,
      operator_note,
    }),
    {
      onSuccess: (response) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-decision-trace']);
        queryClient.invalidateQueries(['agent-decision-trace-analytics']);
        const successfulIds = new Set(
          response.results
            .filter((row) => row.ok)
            .map((row) => String(row.profile_opportunity_id || row.portfolio_opportunity_id || '').trim())
            .filter(Boolean)
        );
        if (successfulIds.size > 0) {
          setQueueSelection((prev) => {
            const next = { ...prev };
            selectedQueueItems.forEach((item) => {
              const opportunityId = String(item.profile_opportunity_id || item.portfolio_opportunity_id || '').trim();
              if (successfulIds.has(opportunityId)) {
                delete next[item.queue_key];
              }
            });
            return next;
          });
        }
        if (response.failed === 0) {
          setQueueBulkNote('');
          toast.success(
            `Bulk follow-up ${response.applied === 1 ? 'action' : 'actions'} applied to ${response.applied} item${response.applied === 1 ? '' : 's'}`
          );
          return;
        }
        const failedLabels = response.results
          .filter((row) => !row.ok)
          .slice(0, 3)
          .map((row) => `${String(row.profile_opportunity_id || row.portfolio_opportunity_id || '').slice(0, 20)}: ${row.error || 'failed'}`);
        toast.error(`Applied ${response.applied}/${response.requested_count}. ${failedLabels.join(' | ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Bulk follow-up action failed');
      },
    }
  );

  const bulkInboxFollowUpActionMutation = useMutation(
    ({
      domain_research_profile_id,
      profile_opportunity_ids,
      portfolio_id,
      portfolio_opportunity_ids,
      action,
      operator_note,
    }: {
      domain_research_profile_id?: string;
      profile_opportunity_ids?: string[];
      portfolio_id?: string;
      portfolio_opportunity_ids?: string[];
      action: 'approve_launch' | 'reject_launch';
      operator_note?: string;
    }) => apiClient.bulkActionAgentCheckpointQueueFollowUp({
      domain_research_profile_id,
      profile_opportunity_ids,
      portfolio_id,
      portfolio_opportunity_ids,
      action,
      operator_note,
    }),
    {
      onSuccess: (response) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-decision-trace']);
        queryClient.invalidateQueries(['agent-decision-trace-analytics']);
        queryClient.invalidateQueries(['notifications']);
        queryClient.invalidateQueries(['notifications-unread-count']);
        const successfulIds = new Set(
          response.results
            .filter((row) => row.ok)
            .map((row) => String(row.profile_opportunity_id || row.portfolio_opportunity_id || '').trim())
            .filter(Boolean)
        );
        if (successfulIds.size > 0) {
          setSelectedInboxIds((prev) => {
            const next = { ...prev };
            selectedInboxItems.forEach((item) => {
              const opportunityId = String(item.origin_opportunity_id || '').trim();
              if (successfulIds.has(opportunityId)) {
                delete next[String(item.id)];
              }
            });
            return next;
          });
        }
        if (response.failed === 0) {
          setInboxBulkFollowUpNote('');
          toast.success(
            `Bulk follow-up ${response.applied === 1 ? 'action' : 'actions'} applied to ${response.applied} item${response.applied === 1 ? '' : 's'}`
          );
          return;
        }
        const failedLabels = response.results
          .filter((row) => !row.ok)
          .slice(0, 3)
          .map((row) => `${String(row.profile_opportunity_id || row.portfolio_opportunity_id || '').slice(0, 20)}: ${row.error || 'failed'}`);
        toast.error(`Applied ${response.applied}/${response.requested_count}. ${failedLabels.join(' | ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Bulk follow-up action failed');
      },
    }
  );

  const bulkInboxRelaunchMutation = useMutation(
    ({ item_ids, operator_note }: { item_ids: string[]; operator_note?: string }) =>
      apiClient.bulkRelaunchInboxFollowUp({ item_ids, operator_note }),
    {
      onSuccess: (response) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-decision-trace']);
        queryClient.invalidateQueries(['agent-decision-trace-analytics']);
        queryClient.invalidateQueries(['notifications']);
        queryClient.invalidateQueries(['notifications-unread-count']);
        const successfulIds = new Set(
          response.results
            .filter((row) => row.ok)
            .map((row) => String(row.item_id || '').trim())
            .filter(Boolean)
        );
        if (successfulIds.size > 0) {
          setSelectedInboxIds((prev) => {
            const next = { ...prev };
            selectedInboxItems.forEach((item) => {
              if (successfulIds.has(String(item.id))) {
                delete next[String(item.id)];
              }
            });
            return next;
          });
        }
        if (response.failed === 0) {
          setInboxBulkFollowUpNote('');
          toast.success(
            `Bulk relaunch applied to ${response.applied} item${response.applied === 1 ? '' : 's'}`
          );
          return;
        }
        const failedLabels = response.results
          .filter((row) => !row.ok)
          .slice(0, 3)
          .map((row) => `${String(row.item_id || '').slice(0, 20)}: ${row.error || 'failed'}`);
        toast.error(`Applied ${response.applied}/${response.requested_count}. ${failedLabels.join(' | ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Bulk relaunch failed');
      },
    }
  );

  const buildInlineFollowUpReviewKey = useCallback(
    (scope: 'domain' | 'fleet', ownerId: string, row: Record<string, any>, idx: number) =>
      `${scope}:${String(ownerId)}:${String(row.opportunity_id || row.canonical_key || idx)}`,
    []
  );

  const buildBulkFollowUpOwnerKey = useCallback(
    (scope: 'domain' | 'fleet', ownerId: string) => `${scope}:${String(ownerId)}`,
    []
  );

  const buildBulkFollowUpSelectionKey = useCallback(
    (scope: 'domain' | 'fleet', ownerId: string, opportunityId: string) =>
      `${buildBulkFollowUpOwnerKey(scope, ownerId)}:${String(opportunityId)}`,
    [buildBulkFollowUpOwnerKey]
  );

  const renderInlineFollowUpApprovalRow = useCallback(
    (
      scope: 'domain' | 'fleet',
      ownerId: string,
      row: Record<string, any>,
      idx: number,
    ) => {
      const opportunityId = String(row.opportunity_id || '').trim();
      const ownerKey = buildBulkFollowUpOwnerKey(scope, ownerId);
      const selectionKey = buildBulkFollowUpSelectionKey(scope, ownerId, opportunityId);
      const reviewKey = buildInlineFollowUpReviewKey(scope, ownerId, row, idx);
      const noteValue = followUpReviewNoteDrafts[reviewKey] || '';
      const isSubmitting = followUpQueueActionMutation.isLoading && activeFollowUpReviewKey === reviewKey;
      const isBulkSubmitting = bulkFollowUpQueueActionMutation.isLoading && activeBulkFollowUpOwnerKey === ownerKey;
      const isSelected = Boolean(bulkFollowUpSelection[selectionKey]);
      const missingIdentifiers = !ownerId || !opportunityId;
      const launchStatus = String(row.follow_up_launch_status || row.follow_up_review_status || '').trim();
      const childJobId = String(row.follow_up_job_id || row.child_job_id || '').trim();
      const submitAction = (action: 'approve_launch' | 'reject_launch') => {
        if (missingIdentifiers) {
          toast.error('Missing follow-up approval identifiers');
          return;
        }
        followUpQueueActionMutation.mutate({
          domain_research_profile_id: scope === 'domain' ? ownerId : undefined,
          profile_opportunity_id: scope === 'domain' ? opportunityId : undefined,
          portfolio_id: scope === 'fleet' ? ownerId : undefined,
          portfolio_opportunity_id: scope === 'fleet' ? opportunityId : undefined,
          action,
          operator_note: noteValue.trim() || undefined,
          navigateOnLaunch: false,
          refreshTarget: scope,
          reviewRowKey: reviewKey,
        });
      };
      return renderAutonomySummaryRow(
        scope,
        ownerId,
        'pending',
        row,
        idx,
        <div className="rounded border border-gray-200 p-2">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex items-start gap-2">
              <input
                type="checkbox"
                className="mt-1 rounded border-gray-300"
                aria-label={`Select ${String(row.title || row.canonical_key || 'opportunity')}`}
                checked={isSelected}
                disabled={missingIdentifiers || isSubmitting || isBulkSubmitting}
                onChange={() => {
                  if (missingIdentifiers) return;
                  setBulkFollowUpSelection((prev) => {
                    const next = { ...prev };
                    if (next[selectionKey]) {
                      delete next[selectionKey];
                    } else {
                      next[selectionKey] = true;
                    }
                    return next;
                  });
                }}
              />
              <div className="min-w-0">
                <div className="text-gray-800">
                  {String(row.title || row.canonical_key || 'Opportunity')}
                  {row.reason_code ? ` · ${String(row.reason_code).replaceAll('_', ' ')}` : ''}
                </div>
                {(launchStatus || childJobId) ? (
                  <div className="mt-1 text-xs text-gray-500">
                    {launchStatus ? `State ${launchStatus.replaceAll('_', ' ')}` : null}
                    {launchStatus && childJobId ? ' · ' : null}
                    {childJobId ? `Job ${childJobId}` : null}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Button
                size="sm"
                variant="primary"
                disabled={isSubmitting || isBulkSubmitting || missingIdentifiers}
                onClick={() => submitAction('approve_launch')}
              >
                <ThumbsUp className="w-4 h-4 mr-1" />
                Approve
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={isSubmitting || isBulkSubmitting || missingIdentifiers}
                onClick={() => submitAction('reject_launch')}
              >
                <ThumbsDown className="w-4 h-4 mr-1" />
                Reject
              </Button>
            </div>
          </div>
          <textarea
            aria-label={`Operator note for ${String(row.title || row.canonical_key || 'opportunity')}`}
            className="mt-2 w-full border border-gray-300 rounded px-2 py-1 text-xs"
            rows={2}
            placeholder="Operator note (optional)"
            value={noteValue}
            disabled={isSubmitting || isBulkSubmitting || missingIdentifiers}
            onChange={(e) => setFollowUpReviewNoteDrafts((prev) => ({ ...prev, [reviewKey]: e.target.value }))}
          />
          {missingIdentifiers ? (
            <div className="mt-1 text-[11px] text-rose-700">Missing identifiers for follow-up approval.</div>
          ) : null}
        </div>
      );
    },
    [
      activeBulkFollowUpOwnerKey,
      activeFollowUpReviewKey,
      buildBulkFollowUpOwnerKey,
      buildBulkFollowUpSelectionKey,
      buildInlineFollowUpReviewKey,
      bulkFollowUpQueueActionMutation,
      bulkFollowUpSelection,
      followUpQueueActionMutation,
      followUpReviewNoteDrafts,
      renderAutonomySummaryRow,
    ]
  );

  const resolveManualBulkFollowUpAction = useCallback(
    (
      row: Record<string, any>,
      opportunities: Array<Record<string, any>> | undefined,
    ): { opportunity: Record<string, any>; action: 'launch_follow_up' | 'relaunch_follow_up' } | null => {
      const opportunityId = String(row.opportunity_id || '').trim();
      if (!opportunityId) return null;
      const matchingOpportunity = (opportunities || []).find(
        (opportunity) => String(opportunity.opportunity_id || '').trim() === opportunityId
      ) as Record<string, any> | undefined;
      if (!matchingOpportunity) return null;
      if (canRelaunchOpportunityRow(matchingOpportunity)) {
        return { opportunity: matchingOpportunity, action: 'relaunch_follow_up' };
      }
      const hasChildJobs = Array.isArray(matchingOpportunity.child_job_ids) && matchingOpportunity.child_job_ids.length > 0;
      if (hasChildJobs) return null;
      return { opportunity: matchingOpportunity, action: 'launch_follow_up' };
    },
    []
  );

  const renderInlineManualRecommendationRow = useCallback(
    (
      scope: 'domain' | 'fleet',
      ownerId: string,
      row: Record<string, any>,
      idx: number,
      opportunities: Array<Record<string, any>> | undefined,
    ) => {
      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
      const bulkAction = resolveManualBulkFollowUpAction(row, opportunities);
      const opportunity = bulkAction?.opportunity;
      const opportunityId = String(opportunity?.opportunity_id || '').trim();
      const ownerKey = buildBulkFollowUpOwnerKey(scope, ownerId);
      const selectionKey = opportunityId ? buildBulkFollowUpSelectionKey(scope, ownerId, opportunityId) : '';
      const isSelected = selectionKey ? Boolean(bulkFollowUpSelection[selectionKey]) : false;
      const isBulkSubmitting = (
        (bulkFollowUpQueueActionMutation.isLoading || bulkManualFollowUpActionMutation.isLoading)
        && activeBulkFollowUpOwnerKey === ownerKey
      );
      return renderAutonomySummaryRow(
        scope,
        ownerId,
        'manual',
        row,
        idx,
        <>
          <div className="flex items-start gap-2">
            {bulkAction && opportunityId ? (
              <input
                type="checkbox"
                className="mt-1 rounded border-gray-300"
                aria-label={`Select ${String(row.title || row.canonical_key || 'manual recommendation')}`}
                checked={isSelected}
                disabled={isBulkSubmitting}
                onChange={() => {
                  setBulkFollowUpSelection((prev) => {
                    const next = { ...prev };
                    if (next[selectionKey]) {
                      delete next[selectionKey];
                    } else {
                      next[selectionKey] = true;
                    }
                    return next;
                  });
                }}
              />
            ) : null}
            <div className="min-w-0">
              <div>{String(row.title || row.canonical_key || 'Manual recommendation')}{row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
              {bulkAction ? (
                <div className="mt-1 text-[11px] text-slate-500">
                  Bulk action {bulkAction.action === 'relaunch_follow_up' ? 'relaunch' : 'launch'} ready
                </div>
              ) : null}
            </div>
          </div>
          {opportunity ? renderManualRecommendationAction(scope, ownerId, opportunity) : null}
          {renderOpportunityExplainabilityPanel(
            buildAutonomyReviewRowKey(scope, ownerId, 'manual', String(row.opportunity_id || row.canonical_key || idx)),
            resolvedRow,
            { surface: scope, ownerId: String(ownerId) }
          )}
        </>
      );
    },
    [
      activeBulkFollowUpOwnerKey,
      buildBulkFollowUpOwnerKey,
      buildBulkFollowUpSelectionKey,
      bulkFollowUpQueueActionMutation.isLoading,
      bulkFollowUpSelection,
      bulkManualFollowUpActionMutation.isLoading,
      renderAutonomySummaryRow,
      renderManualRecommendationAction,
      renderOpportunityExplainabilityPanel,
      buildAutonomyReviewRowKey,
      resolveManualBulkFollowUpAction,
      resolveOpportunityContextRow,
    ]
  );

  const resolveSuppressedBulkRelaunchAction = useCallback(
    (
      row: Record<string, any>,
      opportunities: Array<Record<string, any>> | undefined,
    ): { opportunity: Record<string, any>; action: 'relaunch_follow_up' } | null => {
      const opportunityId = String(row.opportunity_id || '').trim();
      if (!opportunityId) return null;
      const matchingOpportunity = (opportunities || []).find(
        (opportunity) => String(opportunity.opportunity_id || '').trim() === opportunityId
      ) as Record<string, any> | undefined;
      if (!matchingOpportunity || !canRelaunchOpportunityRow(matchingOpportunity)) return null;
      return { opportunity: matchingOpportunity, action: 'relaunch_follow_up' };
    },
    []
  );

  const renderInlineSuppressedRelaunchRow = useCallback(
    (
      scope: 'domain' | 'fleet',
      ownerId: string,
      row: Record<string, any>,
      idx: number,
      opportunities: Array<Record<string, any>> | undefined,
    ) => {
      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
      const bulkAction = resolveSuppressedBulkRelaunchAction(row, opportunities);
      const matchingOpportunity = bulkAction?.opportunity;
      const canRelaunch = Boolean(bulkAction && matchingOpportunity);
      const opportunityId = String(matchingOpportunity?.opportunity_id || '').trim();
      const ownerKey = buildBulkFollowUpOwnerKey(scope, ownerId);
      const selectionKey = opportunityId ? buildBulkFollowUpSelectionKey(scope, ownerId, opportunityId) : '';
      const isSelected = selectionKey ? Boolean(bulkFollowUpSelection[selectionKey]) : false;
      const isBulkSubmitting = (
        (bulkFollowUpQueueActionMutation.isLoading || bulkManualFollowUpActionMutation.isLoading)
        && activeBulkFollowUpOwnerKey === ownerKey
      );
      const isDraftOpen = canRelaunch
        && opportunityNoteDraft?.mode === 'relaunch'
        && opportunityNoteDraft.surface === scope
        && String(opportunityNoteDraft.ownerId) === String(ownerId)
        && String(opportunityNoteDraft.opportunityId) === String(matchingOpportunity?.opportunity_id || '');
      const labelPrefix = scope === 'fleet' ? 'Fleet' : 'Domain';
      return renderAutonomySummaryRow(
        scope,
        ownerId,
        'suppressed',
        row,
        idx,
        <>
          <div className="flex items-start gap-2">
            {canRelaunch && opportunityId ? (
              <input
                type="checkbox"
                className="mt-1 rounded border-gray-300"
                aria-label={`Select ${String(row.title || row.canonical_key || 'suppressed relaunch')}`}
                checked={isSelected}
                disabled={isBulkSubmitting}
                onChange={() => {
                  setBulkFollowUpSelection((prev) => {
                    const next = { ...prev };
                    if (next[selectionKey]) {
                      delete next[selectionKey];
                    } else {
                      next[selectionKey] = true;
                    }
                    return next;
                  });
                }}
              />
            ) : null}
            <div className="min-w-0">
              <div>{String(row.title || row.canonical_key || 'Suppressed relaunch')} · {String(row.reason_code || 'suppressed').replaceAll('_', ' ')}</div>
              {canRelaunch ? (
                <div className="mt-1 text-[11px] text-slate-500">
                  Bulk action relaunch ready
                </div>
              ) : null}
            </div>
          </div>
          {canRelaunch ? (
            <div className="mt-2">
              {isDraftOpen ? (
                <div className="rounded border border-emerald-200 bg-emerald-50 p-2">
                  <div className="text-[11px] font-medium text-emerald-700">Relaunch note</div>
                  <textarea
                    aria-label={`${labelPrefix} relaunch note`}
                    className="mt-2 w-full border border-emerald-200 rounded px-2 py-1 text-xs"
                    rows={3}
                    value={opportunityNoteDraft?.value || ''}
                    onChange={(e) => setOpportunityNoteDraft((prev) => prev ? { ...prev, value: e.target.value } : prev)}
                  />
                  <div className="mt-2 flex gap-2">
                    <Button size="sm" variant="secondary" onClick={submitOpportunityAction}>
                      Relaunch follow-up
                    </Button>
                    <Button size="sm" variant="ghost" onClick={cancelOpportunityAction}>
                      Cancel
                    </Button>
                  </div>
                </div>
              ) : (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    if (!matchingOpportunity) return;
                    beginOpportunityRelaunch(scope, ownerId, matchingOpportunity);
                  }}
                >
                  Relaunch Follow-up
                </Button>
              )}
            </div>
          ) : null}
          {renderOpportunityExplainabilityPanel(
            buildAutonomyReviewRowKey(scope, ownerId, 'suppressed', String(row.opportunity_id || row.canonical_key || idx)),
            resolvedRow,
            { surface: scope, ownerId: String(ownerId) }
          )}
        </>
      );
    },
    [
      activeBulkFollowUpOwnerKey,
      beginOpportunityRelaunch,
      buildBulkFollowUpOwnerKey,
      buildBulkFollowUpSelectionKey,
      bulkFollowUpQueueActionMutation.isLoading,
      bulkFollowUpSelection,
      bulkManualFollowUpActionMutation.isLoading,
      cancelOpportunityAction,
      opportunityNoteDraft,
      renderAutonomySummaryRow,
      renderOpportunityExplainabilityPanel,
      buildAutonomyReviewRowKey,
      resolveSuppressedBulkRelaunchAction,
      resolveOpportunityContextRow,
      submitOpportunityAction,
    ]
  );

  const renderBulkFollowUpControls = useCallback(
    (
      scope: 'domain' | 'fleet',
      ownerId: string,
      approvalRows: Array<Record<string, any>> | undefined,
      manualRows?: Array<Record<string, any>> | undefined,
      suppressedRows?: Array<Record<string, any>> | undefined,
      opportunities?: Array<Record<string, any>> | undefined,
    ) => {
      const availableApprovalRows = (approvalRows || []).filter((row) => String(row.opportunity_id || '').trim());
      const availableActionRows = [
        ...(manualRows || []).map((row) => ({ kind: 'manual' as const, row })),
        ...(suppressedRows || []).map((row) => ({ kind: 'suppressed' as const, row })),
      ]
        .map((row) => {
          const resolved = row.kind === 'suppressed'
            ? resolveSuppressedBulkRelaunchAction(row.row, opportunities)
            : resolveManualBulkFollowUpAction(row.row, opportunities);
          if (!resolved) return null;
          return {
            row: row.row,
            opportunityId: String(resolved.opportunity.opportunity_id || '').trim(),
            action: resolved.action,
            kind: row.kind,
          };
        })
        .filter(Boolean) as Array<{
          row: Record<string, any>;
          opportunityId: string;
          action: 'launch_follow_up' | 'relaunch_follow_up';
          kind: 'manual' | 'suppressed';
        }>;
      const totalSelectable = availableApprovalRows.length + availableActionRows.length;
      if (totalSelectable === 0) return null;
      const ownerKey = buildBulkFollowUpOwnerKey(scope, ownerId);
      const selectedApprovalIds = availableApprovalRows
        .map((row) => String(row.opportunity_id || '').trim())
        .filter((opportunityId) => Boolean(bulkFollowUpSelection[buildBulkFollowUpSelectionKey(scope, ownerId, opportunityId)]));
      const selectedActionRows = availableActionRows.filter((row) =>
        Boolean(bulkFollowUpSelection[buildBulkFollowUpSelectionKey(scope, ownerId, row.opportunityId)])
      );
      const selectedActionModes = Array.from(new Set(selectedActionRows.map((row) => row.action)));
      const selectedOpportunityIds = selectedApprovalIds.length > 0
        ? selectedApprovalIds
        : selectedActionRows.map((row) => row.opportunityId);
      const noteValue = bulkFollowUpNotes[ownerKey] || '';
      const isSubmitting = (
        (bulkFollowUpQueueActionMutation.isLoading || bulkManualFollowUpActionMutation.isLoading)
        && activeBulkFollowUpOwnerKey === ownerKey
      );
      const selectedCount = selectedApprovalIds.length + selectedActionRows.length;
      const mixedSelection = selectedApprovalIds.length > 0 && selectedActionRows.length > 0;
      const mixedActionModes = selectedActionModes.length > 1;
      const disabledReason = mixedSelection
        ? 'Bulk follow-up actions cannot mix pending approvals and launch/relaunch selections.'
        : mixedActionModes
          ? 'Bulk manual follow-up actions cannot mix launch and relaunch selections.'
          : '';
      const submitBulkAction = (action: 'approve_launch' | 'reject_launch') => {
        if (selectedApprovalIds.length === 0) {
          toast.error('Select at least one follow-up approval');
          return;
        }
        bulkFollowUpQueueActionMutation.mutate({
          domain_research_profile_id: scope === 'domain' ? ownerId : undefined,
          profile_opportunity_ids: scope === 'domain' ? selectedOpportunityIds : undefined,
          portfolio_id: scope === 'fleet' ? ownerId : undefined,
          portfolio_opportunity_ids: scope === 'fleet' ? selectedOpportunityIds : undefined,
          action,
          operator_note: noteValue.trim() || undefined,
          ownerKey,
          refreshTarget: scope,
        });
      };
      const submitBulkManualAction = (action: 'launch_follow_up' | 'relaunch_follow_up') => {
        if (selectedActionRows.length === 0) {
          toast.error(`Select at least one follow-up to ${action === 'relaunch_follow_up' ? 'relaunch' : 'launch'}`);
          return;
        }
        bulkManualFollowUpActionMutation.mutate({
          scope,
          ownerId,
          opportunityIds: selectedActionRows.map((row) => row.opportunityId),
          action,
          operator_note: noteValue.trim() || undefined,
        });
      };
      return (
        <div className="rounded border border-slate-200 bg-slate-50 p-2 space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
            <Button
              size="sm"
              variant="ghost"
              disabled={isSubmitting}
              onClick={() => {
                setBulkFollowUpSelection((prev) => {
                  const next = { ...prev };
                  availableApprovalRows.forEach((row) => {
                    next[buildBulkFollowUpSelectionKey(scope, ownerId, String(row.opportunity_id || '').trim())] = true;
                  });
                  availableActionRows.forEach((row) => {
                    next[buildBulkFollowUpSelectionKey(scope, ownerId, row.opportunityId)] = true;
                  });
                  return next;
                });
              }}
            >
              Select all
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={isSubmitting}
              onClick={() => {
                setBulkFollowUpSelection((prev) => {
                  const next = { ...prev };
                  availableApprovalRows.forEach((row) => {
                    delete next[buildBulkFollowUpSelectionKey(scope, ownerId, String(row.opportunity_id || '').trim())];
                  });
                  availableActionRows.forEach((row) => {
                    delete next[buildBulkFollowUpSelectionKey(scope, ownerId, row.opportunityId)];
                  });
                  return next;
                });
              }}
            >
              Clear
            </Button>
            <span>Selected {selectedCount} of {totalSelectable}</span>
          </div>
          <textarea
            className="w-full border border-gray-300 rounded px-2 py-1 text-xs"
            rows={2}
            placeholder="Shared operator note (optional)"
            value={noteValue}
            disabled={isSubmitting}
            onChange={(e) => setBulkFollowUpNotes((prev) => ({ ...prev, [ownerKey]: e.target.value }))}
          />
          {disabledReason ? (
            <div className="text-[11px] text-amber-700">{disabledReason}</div>
          ) : null}
          <div className="flex flex-wrap gap-2">
            {selectedApprovalIds.length > 0 || (selectedCount === 0 && availableApprovalRows.length > 0) ? (
              <>
                <Button size="sm" variant="primary" disabled={isSubmitting || selectedApprovalIds.length === 0 || Boolean(disabledReason)} onClick={() => submitBulkAction('approve_launch')}>
                  Approve Selected
                </Button>
                <Button size="sm" variant="ghost" disabled={isSubmitting || selectedApprovalIds.length === 0 || Boolean(disabledReason)} onClick={() => submitBulkAction('reject_launch')}>
                  Reject Selected
                </Button>
              </>
            ) : null}
            {selectedActionRows.length > 0 || (selectedCount === 0 && availableActionRows.length > 0) ? (
              <Button
                size="sm"
                variant="secondary"
                disabled={isSubmitting || selectedActionRows.length === 0 || Boolean(disabledReason)}
                onClick={() => submitBulkManualAction(selectedActionModes[0] || 'launch_follow_up')}
              >
                {(selectedActionModes[0] || 'launch_follow_up') === 'relaunch_follow_up' ? 'Relaunch Selected' : 'Launch Selected'}
              </Button>
            ) : null}
          </div>
        </div>
      );
    },
    [
      activeBulkFollowUpOwnerKey,
      buildBulkFollowUpOwnerKey,
      buildBulkFollowUpSelectionKey,
      bulkFollowUpNotes,
      bulkFollowUpQueueActionMutation,
      bulkFollowUpSelection,
      bulkManualFollowUpActionMutation,
      resolveManualBulkFollowUpAction,
      resolveSuppressedBulkRelaunchAction,
    ]
  );

  const quickStartClaudeBackendMutation = useMutation(
    (data: AgentJobQuickStartClaudeBackendRequest) => apiClient.quickStartClaudeBackendJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Claude backend loop started');
        setShowClaudeQuickStartModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        const detail = error?.response?.data?.detail;
        if (detail && typeof detail === 'object' && Array.isArray((detail as any).blocked_commands)) {
          const blocked = ((detail as any).blocked_commands as any[])
            .map((x) => String(x || '').trim())
            .filter(Boolean)
            .slice(0, 3);
          toast.error(
            blocked.length > 0
              ? `Blocked unsafe command(s): ${blocked.join(' | ')}`
              : String((detail as any).message || 'Blocked unsafe command(s)')
          );
          return;
        }
        toast.error(
          (typeof detail === 'string' ? detail : '') ||
            error?.message ||
            'Failed to start Claude backend loop'
        );
      },
    }
  );

  const quickStartDomainResearchMutation = useMutation(
    (data: AgentJobQuickStartDomainResearchRequest) => apiClient.quickStartDomainResearchJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Domain research started');
        setShowDomainResearchQuickStartModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        const detail = error?.response?.data?.detail;
        toast.error(
          (typeof detail === 'string' ? detail : '') ||
            error?.message ||
            'Failed to start domain research'
        );
      },
    }
  );

  const quickStartRepoBugTriageMutation = useMutation(
    (data: AgentJobQuickStartRepoBugTriageRequest) => apiClient.quickStartRepoBugTriageJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Repo bug triage started');
        setShowRepoBugTriageQuickStartModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        const detail = error?.response?.data?.detail;
        if (detail && typeof detail === 'object' && Array.isArray((detail as any).blocked_commands)) {
          const blocked = ((detail as any).blocked_commands as any[])
            .map((x) => String(x || '').trim())
            .filter(Boolean)
            .slice(0, 3);
          toast.error(
            blocked.length > 0
              ? `Blocked unsafe command(s): ${blocked.join(' | ')}`
              : String((detail as any).message || 'Blocked unsafe command(s)')
          );
          return;
        }
        toast.error(
          (typeof detail === 'string' ? detail : '') ||
            error?.message ||
            'Failed to start repo bug triage'
        );
      },
    }
  );

  const quickStartBugTriageSwarmMutation = useMutation(
    (data: AgentJobQuickStartBugTriageSwarmRequest) => apiClient.quickStartBugTriageSwarmJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Bug triage swarm started');
        setShowBugTriageSwarmQuickStartModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        const detail = error?.response?.data?.detail;
        if (detail && typeof detail === 'object' && Array.isArray((detail as any).blocked_commands)) {
          const blocked = ((detail as any).blocked_commands as any[])
            .map((x) => String(x || '').trim())
            .filter(Boolean)
            .slice(0, 3);
          toast.error(
            blocked.length > 0
              ? `Blocked unsafe command(s): ${blocked.join(' | ')}`
              : String((detail as any).message || 'Blocked unsafe command(s)')
          );
          return;
        }
        toast.error(
          (typeof detail === 'string' ? detail : '') ||
            error?.message ||
            'Failed to start bug triage swarm'
        );
      },
    }
  );

  const quickStartBuildBreakSwarmMutation = useMutation(
    (data: AgentJobQuickStartBuildBreakSwarmRequest) => apiClient.quickStartBuildBreakSwarmJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Build break swarm started');
        setShowBuildBreakSwarmQuickStartModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        const detail = error?.response?.data?.detail;
        if (detail && typeof detail === 'object' && Array.isArray((detail as any).blocked_commands)) {
          const blocked = ((detail as any).blocked_commands as any[])
            .map((x) => String(x || '').trim())
            .filter(Boolean)
            .slice(0, 3);
          toast.error(
            blocked.length > 0
              ? `Blocked unsafe command(s): ${blocked.join(' | ')}`
              : String((detail as any).message || 'Blocked unsafe command(s)')
          );
          return;
        }
        toast.error((typeof detail === 'string' ? detail : '') || error?.message || 'Failed to start build break swarm');
      },
    }
  );

  const quickStartFrontendRegressionSwarmMutation = useMutation(
    (data: AgentJobQuickStartFrontendRegressionSwarmRequest) => apiClient.quickStartFrontendRegressionSwarmJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Frontend regression swarm started');
        setShowFrontendRegressionSwarmQuickStartModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        const detail = error?.response?.data?.detail;
        if (detail && typeof detail === 'object' && Array.isArray((detail as any).blocked_commands)) {
          const blocked = ((detail as any).blocked_commands as any[])
            .map((x) => String(x || '').trim())
            .filter(Boolean)
            .slice(0, 3);
          toast.error(
            blocked.length > 0
              ? `Blocked unsafe command(s): ${blocked.join(' | ')}`
              : String((detail as any).message || 'Blocked unsafe command(s)')
          );
          return;
        }
        toast.error(
          (typeof detail === 'string' ? detail : '') || error?.message || 'Failed to start frontend regression swarm'
        );
      },
    }
  );

  /**
   * Which coding-swarm quick start is open, if any.
   *
   * These were three components whose only difference was their text, colour
   * and mutation. The differences live in SWARM_QUICK_START_PRESETS now; what
   * is left here is the part that genuinely belongs to this component — which
   * flag is set, which mutation to call, and how to close.
   */
  const activeSwarmQuickStart = useMemo(() => {
    const open = showBugTriageSwarmQuickStartModal
      ? {
          presetKey: 'bug_triage_swarm',
          mutation: quickStartBugTriageSwarmMutation,
          buildPayload: buildBugTriageSwarmQuickStartPayload,
          setOpen: setShowBugTriageSwarmQuickStartModal,
        }
      : showBuildBreakSwarmQuickStartModal
        ? {
            presetKey: 'build_break_swarm',
            mutation: quickStartBuildBreakSwarmMutation,
            buildPayload: buildBuildBreakSwarmQuickStartPayload,
            setOpen: setShowBuildBreakSwarmQuickStartModal,
          }
        : showFrontendRegressionSwarmQuickStartModal
          ? {
              presetKey: 'frontend_regression_swarm',
              mutation: quickStartFrontendRegressionSwarmMutation,
              buildPayload: buildFrontendRegressionSwarmQuickStartPayload,
              setOpen: setShowFrontendRegressionSwarmQuickStartModal,
            }
          : null;
    if (!open) return null;
    const preset = swarmQuickStartPreset(open.presetKey);
    return preset ? { ...open, preset } : null;
  }, [
    showBugTriageSwarmQuickStartModal,
    showBuildBreakSwarmQuickStartModal,
    showFrontendRegressionSwarmQuickStartModal,
    quickStartBugTriageSwarmMutation,
    quickStartBuildBreakSwarmMutation,
    quickStartFrontendRegressionSwarmMutation,
  ]);


  const quickStartRoleWorkflowMutation = useMutation(
    (data: AgentJobQuickStartRoleWorkflowRequest) => apiClient.quickStartRoleWorkflowJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Role workflow started');
        setShowRoleWorkflowQuickStartModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        const detail = error?.response?.data?.detail;
        toast.error(
          (typeof detail === 'string' ? detail : '') ||
            error?.message ||
            'Failed to start role workflow'
        );
      },
    }
  );

  const [paperAlgoRunDemo, setPaperAlgoRunDemo] = useState<Record<string, boolean>>({});
  const [paperAlgoEntrypoint, setPaperAlgoEntrypoint] = useState<Record<string, string>>({});
  const [paperAlgoEntrypointSavedAt, setPaperAlgoEntrypointSavedAt] = useState<Record<string, string>>({});
  const [paperAlgoEntrypointSaving, setPaperAlgoEntrypointSaving] = useState<Record<string, boolean>>({});
  const [paperAlgoEntrypointError, setPaperAlgoEntrypointError] = useState<Record<string, string>>({});

  const { data: myPreferences } = useQuery(['me-preferences'], () => apiClient.getMyPreferences(), {
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });
  const updateMyPreferencesMutation = useMutation((updates: any) => apiClient.updateMyPreferences(updates), {
    onSuccess: () => {
      queryClient.invalidateQueries(['me-preferences']);
      toast.success('Preferences updated');
    },
    onError: (e: any) => {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to update preferences');
    },
  });
  const paperAlgoDefaultRunDemoCheck = (myPreferences as any)?.paper_algo_default_run_demo_check === true;

  const { data: unsafeExecAvailability } = useQuery(
    ['unsafe-exec-availability'],
    () => apiClient.getUnsafeExecAvailability(),
    { staleTime: 30_000, refetchOnWindowFocus: false }
  );

  const unsafeExecBadge = useMemo(() => {
    const avail: any = unsafeExecAvailability as any;
    const enabled = !!avail?.enabled;
    const backend = String(avail?.backend || 'subprocess');
    const dockerOk = backend !== 'docker' || (avail?.docker?.available === true && avail?.docker?.image_present === true);
    const status: 'ready' | 'blocked' | 'off' = enabled && dockerOk ? 'ready' : enabled ? 'blocked' : 'off';
    const label =
      status === 'ready'
        ? 'demo-check ready'
        : status === 'blocked'
          ? 'demo-check not ready'
          : 'demo-check off';
    const title =
      status === 'ready'
        ? `Behavioral demo check available (backend: ${backend})`
        : status === 'blocked'
          ? `Behavioral demo check enabled but not ready (backend: ${backend})`
          : 'Behavioral demo check disabled on server';
    const color =
      status === 'ready' ? 'bg-green-500' : status === 'blocked' ? 'bg-amber-500' : 'bg-gray-400';
    return { status, label, title, color };
  }, [unsafeExecAvailability]);

  const paperAlgoDefaultToggleTitle =
    unsafeExecBadge.status === 'ready'
      ? 'Set the default for new items in this session'
      : 'Server not ready for demo checks (see badge)';

  const normalizeEntrypoint = (raw: string): { ok: boolean; value: string; error?: string } => {
    let v = String(raw || '').trim();
    if (!v) return { ok: true, value: 'demo.py' };
    v = v.replace(/\\/g, '/');
    while (v.startsWith('./')) v = v.slice(2);
    if (v.startsWith('/') || v.startsWith('~') || v.includes(':')) return { ok: false, value: v, error: 'Absolute paths not allowed' };
    if (v.split('/').some((p) => p === '..')) return { ok: false, value: v, error: "'..' not allowed" };
    if (/\s/.test(v)) return { ok: false, value: v, error: 'Whitespace not allowed' };
    if (!v.endsWith('.py')) return { ok: false, value: v, error: 'Must end with .py' };
    if (!/^[A-Za-z0-9._/\\-]+$/.test(v)) return { ok: false, value: v, error: 'Invalid characters' };
    return { ok: true, value: v };
  };

  // Format time duration
  const formatDuration = (startedAt?: string, completedAt?: string) => {
    if (!startedAt) return '-';
    const start = new Date(startedAt);
    const end = completedAt ? new Date(completedAt) : new Date();
    const diff = Math.floor((end.getTime() - start.getTime()) / 1000);
    if (diff < 60) return `${diff}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ${diff % 60}s`;
    return `${Math.floor(diff / 3600)}h ${Math.floor((diff % 3600) / 60)}m`;
  };

  const getGraphHealthStatus = (job: AgentJob): string => {
    return String((job.results as any)?.execution_strategy?.execution_graph?.graph_health?.status || '').toLowerCase();
  };

  const getGraphHealthSeverity = (job: AgentJob): number => {
    return Number((job.results as any)?.execution_strategy?.execution_graph?.graph_health?.severity_score || 0);
  };

  const getMemoryDedupSkipped = (job: AgentJob): number => {
    return Number(
      (job.results as any)?.execution_strategy?.memory_persistence?.extraction?.skipped_duplicates || 0
    );
  };

  const getScopeGuardBlocks = (job: AgentJob): number => {
    const events = (job.results as any)?.execution_strategy?.scope_observability?.events;
    if (!Array.isArray(events)) return 0;
    return events.filter((event: any) => String(event?.type || '').trim() === 'scope_guard_blocked').length;
  };


  const matchesExperimentRecoveryFilter = useCallback((job: AgentJob, filter: string): boolean => {
    if (!filter) return true;
    const run = getLatestExperimentRun(job);
    if (!run) return false;
    const failedCommands = summarizeExperimentRun(run).failedCommands.length;
    if (filter === 'bootstrap_attempted') return Boolean(run.bootstrap_attempted);
    if (filter === 'bootstrap_recovered') return Boolean(run.bootstrap_attempted && run.bootstrap_ok);
    if (filter === 'fallback_attempted') return Boolean(run.fallback_attempted);
    if (filter === 'fallback_ok') return Boolean(run.fallback_attempted && run.fallback_ok);
    if (filter === 'unresolved_recovery') return Boolean(failedCommands > 0 && run.fallback_attempted && !run.fallback_ok);
    return true;
  }, []);

  const getExperimentRecoveryPriority = useCallback((job: AgentJob): number => {
    const run = getLatestExperimentRun(job);
    return getExperimentRecoveryPriorityForRun(run);
  }, []);

  const jobsForDisplay = useMemo(() => {
    const base = Array.isArray((jobsData as any)?.jobs) ? ([...(jobsData as any).jobs] as AgentJob[]) : [];
    let rows = base;
    if (graphHealthFilter) {
      rows = rows.filter((job) => getGraphHealthStatus(job) === graphHealthFilter);
    }
    if (dedupSkipFilter) {
      rows = rows.filter((job) => {
        const skipped = getMemoryDedupSkipped(job);
        if (dedupSkipFilter === 'gt0') return skipped > 0;
        if (dedupSkipFilter === 'gte3') return skipped >= 3;
        if (dedupSkipFilter === 'gte5') return skipped >= 5;
        return true;
      });
    }
    if (scopeGuardFilter) {
      rows = rows.filter((job) => {
        const blocks = getScopeGuardBlocks(job);
        if (scopeGuardFilter === 'blocked') return blocks > 0;
        if (scopeGuardFilter === 'clean') return blocks === 0;
        return true;
      });
    }
    if (experimentRecoveryFilter) {
      rows = rows.filter((job) => matchesExperimentRecoveryFilter(job, experimentRecoveryFilter));
    }

    if (graphSortBy === 'graph_severity_desc') {
      rows.sort((a, b) => {
        const d = getGraphHealthSeverity(b) - getGraphHealthSeverity(a);
        if (d !== 0) return d;
        return new Date(String(b.created_at || '')).getTime() - new Date(String(a.created_at || '')).getTime();
      });
    } else if (graphSortBy === 'graph_health_critical_first') {
      const rank = (s: string): number =>
        s === 'critical' ? 0 : s === 'warning' ? 1 : s === 'unknown' ? 2 : s === 'ok' ? 3 : 4;
      rows.sort((a, b) => {
        const d = rank(getGraphHealthStatus(a)) - rank(getGraphHealthStatus(b));
        if (d !== 0) return d;
        return getGraphHealthSeverity(b) - getGraphHealthSeverity(a);
      });
    } else if (graphSortBy === 'scope_guard_blocked_first') {
      rows.sort((a, b) => {
        const blockDelta = getScopeGuardBlocks(b) - getScopeGuardBlocks(a);
        if (blockDelta !== 0) return blockDelta;
        return getGraphHealthSeverity(b) - getGraphHealthSeverity(a);
      });
    } else if (graphSortBy === 'experiment_recovery_priority') {
      rows.sort((a, b) => {
        const recoveryDelta = getExperimentRecoveryPriority(b) - getExperimentRecoveryPriority(a);
        if (recoveryDelta !== 0) return recoveryDelta;
        return getGraphHealthSeverity(b) - getGraphHealthSeverity(a);
      });
    }

    const pinnedId = String((deepLinkedJobData as any)?.id || '').trim();
    if (pinnedId) {
      const alreadyVisible = rows.some((job) => String(job.id) === pinnedId);
      if (!alreadyVisible) {
        rows = [deepLinkedJobData as AgentJob, ...rows];
      }
    }
    return rows;
  }, [jobsData, graphHealthFilter, dedupSkipFilter, scopeGuardFilter, experimentRecoveryFilter, graphSortBy, deepLinkedJobData, getExperimentRecoveryPriority, matchesExperimentRecoveryFilter]);

  const backlogItems = useMemo(
    () => ((((codingBacklogData as any)?.items || []) as CodingBacklogItem[])),
    [codingBacklogData]
  );
  const filteredBacklogItems = useMemo(() => {
    let rows = [...backlogItems];
    if (backlogAssignmentFilter) {
      rows = rows.filter((item) => String(item.assigned_user_id || '').trim() === backlogAssignmentFilter);
    }
    if (backlogQueueStateFilter) {
      rows = rows.filter((item) => String(item.operator_queue_state || '').trim() === backlogQueueStateFilter);
    }
    rows.sort((a, b) => {
      const priority = (item: CodingBacklogItem) => {
        const state = String(item.operator_queue_state || '').trim();
        if (state === 'new_auto_routed') return 0;
        if (String(item.assigned_user_id || '').trim() === String(user?.id || '').trim()) return 1;
        if (state === 'awaiting_operator_decision') return 2;
        if (state === 'awaiting_assignment') return 3;
        if (state === 'ready_to_start') return 4;
        if (state === 'blocked') return 5;
        if (state === 'superseded') return 6;
        return 7;
      };
      const delta = priority(a) - priority(b);
      if (delta !== 0) return delta;
      return String(b.updated_at || '').localeCompare(String(a.updated_at || ''));
    });
    return rows;
  }, [backlogItems, backlogAssignmentFilter, backlogQueueStateFilter, user]);
  const backlogBySwarmJobId = useMemo(() => {
    const out: Record<string, CodingBacklogItem[]> = {};
    for (const item of backlogItems) {
      const lineage = ((item as any)?.lineage && typeof (item as any).lineage === 'object')
        ? ((item as any).lineage as Record<string, any>)
        : {};
      const swarmJobId = String(lineage.originating_swarm_job_id || '').trim();
      if (!swarmJobId) continue;
      if (!out[swarmJobId]) out[swarmJobId] = [];
      out[swarmJobId].push(item);
    }
    return out;
  }, [backlogItems]);
  const swarmOutcomeCases = useMemo(
    () => ((((swarmOutcomeAnalyticsData as any)?.cases || []) as AgentJobSwarmOutcomeCase[])),
    [swarmOutcomeAnalyticsData]
  );
  const swarmOutcomeBySwarmJobId = useMemo(() => {
    const out: Record<string, AgentJobSwarmOutcomeCase> = {};
    for (const item of swarmOutcomeCases) {
      const key = String(item?.swarm_job_id || '').trim();
      if (key) out[key] = item;
    }
    return out;
  }, [swarmOutcomeCases]);
  const swarmOutcomeByRepairJobId = useMemo(() => {
    const out: Record<string, AgentJobSwarmOutcomeCase> = {};
    for (const item of swarmOutcomeCases) {
      const key = String(item?.repair_job_id || '').trim();
      if (key) out[key] = item;
    }
    return out;
  }, [swarmOutcomeCases]);
  const swarmReviewJobs = useMemo(() => {
    const base = Array.isArray((swarmReviewJobsData as any)?.jobs) ? ((swarmReviewJobsData as any).jobs as AgentJob[]) : [];
    return base.filter((job) => {
      const launchMode = String((job as any)?.launch_mode || ((job.config as any)?.launch_mode || '')).trim().toLowerCase();
      if (!['quick_start_bug_triage_swarm', 'quick_start_build_break_swarm', 'quick_start_frontend_regression_swarm'].includes(launchMode)) {
        return false;
      }
      const swarmSummary = ((job as any)?.swarm_summary && typeof (job as any).swarm_summary === 'object')
        ? ((job as any).swarm_summary as Record<string, any>)
        : null;
      const reviewState = String(swarmSummary?.review_state || '').trim().toLowerCase();
      return ['needs_review', 'insufficient_swarm_consensus', 'consensus_failed', 'tie_break_running', 'manual_promotion'].includes(reviewState)
        || Boolean(swarmSummary?.review_required);
    });
  }, [swarmReviewJobsData]);
  const filteredSwarmReviewJobs = useMemo(() => {
    return swarmReviewJobs.filter((job) => {
      const cfg = (job.config || {}) as Record<string, any>;
      const quickStart = (cfg.quick_start && typeof cfg.quick_start === 'object') ? (cfg.quick_start as Record<string, any>) : {};
      const presetKey = String(quickStart.preset_key || cfg.coding_swarm_preset_key || '').trim().toLowerCase();
      const swarmSummary = (((job as any)?.swarm_summary && typeof (job as any).swarm_summary === 'object')
        ? ((job as any).swarm_summary as Record<string, any>)
        : {}) as Record<string, any>;
      const reviewState = String(swarmSummary.review_state || '').trim().toLowerCase();
      const overallConfidence = Number((swarmSummary.confidence as any)?.overall || 0);
      const confidenceBand = overallConfidence >= 0.7 ? 'high' : overallConfidence >= 0.5 ? 'medium' : 'low';
      const hasBacklog = (backlogBySwarmJobId[String(job.id)] || []).length > 0;
      const assignedUserId = String(swarmSummary.assigned_user_id || '').trim();
      if (swarmReviewPresetFilter && presetKey !== swarmReviewPresetFilter) return false;
      if (swarmReviewStateFilter && reviewState !== swarmReviewStateFilter) return false;
      if (swarmReviewConfidenceBand && confidenceBand !== swarmReviewConfidenceBand) return false;
      if (swarmReviewBacklogFilter === 'linked' && !hasBacklog) return false;
      if (swarmReviewBacklogFilter === 'unlinked' && hasBacklog) return false;
      if (swarmReviewAssignmentFilter === 'assigned_to_me' && assignedUserId !== String(user?.id || '')) return false;
      if (swarmReviewAssignmentFilter === 'unassigned' && assignedUserId) return false;
      if (swarmReviewAssignmentFilter && !['assigned_to_me', 'unassigned'].includes(swarmReviewAssignmentFilter) && assignedUserId !== swarmReviewAssignmentFilter) return false;
      return true;
    });
  }, [swarmReviewJobs, swarmReviewPresetFilter, swarmReviewStateFilter, swarmReviewConfidenceBand, swarmReviewBacklogFilter, swarmReviewAssignmentFilter, backlogBySwarmJobId, user]);

  const jobCountSummary = useMemo(() => {
    const allJobs = Array.isArray((jobsData as any)?.jobs) ? ((jobsData as any).jobs as AgentJob[]) : [];
    const allCount = allJobs.length;
    const shownCount = jobsForDisplay.length;
    const pinnedId = String((deepLinkedJobData as any)?.id || '').trim();
    const pinnedOutsideList = !!pinnedId && !allJobs.some((job) => String((job as any)?.id || '') === pinnedId);
    const pinnedOutsideFilters = !!pinnedId && !allJobs.filter((job) => {
      const graphMatch = !graphHealthFilter || getGraphHealthStatus(job) === graphHealthFilter;
      const dedupSkipped = getMemoryDedupSkipped(job);
      const dedupMatch =
        !dedupSkipFilter ||
        (dedupSkipFilter === 'gt0' && dedupSkipped > 0) ||
        (dedupSkipFilter === 'gte3' && dedupSkipped >= 3) ||
        (dedupSkipFilter === 'gte5' && dedupSkipped >= 5);
      const scopeBlocks = getScopeGuardBlocks(job);
      const scopeGuardMatch =
        !scopeGuardFilter ||
        (scopeGuardFilter === 'blocked' && scopeBlocks > 0) ||
        (scopeGuardFilter === 'clean' && scopeBlocks === 0);
      const experimentMatch = matchesExperimentRecoveryFilter(job, experimentRecoveryFilter);
      return graphMatch && dedupMatch && scopeGuardMatch && experimentMatch;
    }).some((job) => String((job as any)?.id || '') === pinnedId);
    const counts = {
      critical: 0,
      warning: 0,
      ok: 0,
      unknown: 0,
      quick_start_claude_backend: 0,
      quick_start_domain_research: 0,
      quick_start_bug_triage_swarm: 0,
      quick_start_build_break_swarm: 0,
      quick_start_frontend_regression_swarm: 0,
      quick_start_repo_bug_triage: 0,
      quick_start_role_workflow: 0,
      dedup_gt0: 0,
      dedup_gte3: 0,
      scope_guard_blocked: 0,
      bootstrap_attempted: 0,
      bootstrap_recovered: 0,
      fallback_attempted: 0,
      fallback_ok: 0,
      failed_command_total: 0,
      unresolved_failed_command_total: 0,
      unresolved_recovery_jobs: 0,
    };
    for (const job of jobsForDisplay) {
      const s = getGraphHealthStatus(job);
      if (s === 'critical') counts.critical += 1;
      else if (s === 'warning') counts.warning += 1;
      else if (s === 'ok') counts.ok += 1;
      else counts.unknown += 1;
      const launchMode = String((job as any)?.launch_mode || ((job.config as any)?.launch_mode || '')).toLowerCase();
      if (launchMode === 'quick_start_claude_backend') counts.quick_start_claude_backend += 1;
      if (launchMode === 'quick_start_domain_research') counts.quick_start_domain_research += 1;
      if (launchMode === 'quick_start_bug_triage_swarm') counts.quick_start_bug_triage_swarm += 1;
      if (launchMode === 'quick_start_build_break_swarm') counts.quick_start_build_break_swarm += 1;
      if (launchMode === 'quick_start_frontend_regression_swarm') counts.quick_start_frontend_regression_swarm += 1;
      if (launchMode === 'quick_start_repo_bug_triage') counts.quick_start_repo_bug_triage += 1;
      if (launchMode === 'quick_start_role_workflow') counts.quick_start_role_workflow += 1;
      const dedupSkipped = getMemoryDedupSkipped(job);
      if (dedupSkipped > 0) counts.dedup_gt0 += 1;
      if (dedupSkipped >= 3) counts.dedup_gte3 += 1;
      if (getScopeGuardBlocks(job) > 0) counts.scope_guard_blocked += 1;
      const latestRun = getLatestExperimentRun(job);
      if (latestRun?.bootstrap_attempted) counts.bootstrap_attempted += 1;
      if (latestRun?.bootstrap_attempted && latestRun?.bootstrap_ok) counts.bootstrap_recovered += 1;
      if (latestRun?.fallback_attempted) counts.fallback_attempted += 1;
      if (latestRun?.fallback_attempted && latestRun?.fallback_ok) counts.fallback_ok += 1;
      const failedCommands = summarizeExperimentRun(latestRun).failedCommands.length;
      counts.failed_command_total += failedCommands;
      if (failedCommands > 0 && latestRun?.fallback_attempted && !latestRun?.fallback_ok) {
        counts.unresolved_failed_command_total += failedCommands;
        counts.unresolved_recovery_jobs += 1;
      }
    }
    return { allCount, shownCount, pinnedOutsideList, pinnedOutsideFilters, ...counts };
  }, [jobsData, jobsForDisplay, deepLinkedJobData, graphHealthFilter, dedupSkipFilter, scopeGuardFilter, experimentRecoveryFilter, matchesExperimentRecoveryFilter]);

  // Render stats card
  const StatsCard: React.FC<{
    title: string;
    value: string | number;
    icon: React.ComponentType<any>;
    color: string;
    onClick?: () => void;
    titleHint?: string;
    active?: boolean;
  }> = ({
    title,
    value,
    icon: Icon,
    color,
    onClick,
    titleHint,
    active = false,
  }) => (
    <div
      className={`bg-white rounded-lg border p-4 ${
        active ? 'border-indigo-300 ring-1 ring-indigo-200' : 'border-gray-200'
      } ${onClick ? 'cursor-pointer hover:shadow-sm' : ''}`}
      onClick={onClick}
      onKeyDown={(e) => {
        if (!onClick) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      title={titleHint}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500">{title}</p>
          <p className="text-2xl font-semibold mt-1">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );

  // Render job card

  // Render job detail panel

  // Render template card

  // Create job modal
  // Create from template modal
  const renderScientificSandboxManagementPanel = () => (
    <div className="border border-gray-200 rounded-lg p-3 bg-gray-50 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-medium text-gray-800">Scientific Sandboxes</div>
          <div className="text-xs text-gray-500">Stored runtime profiles for recipe-backed scientific validation.</div>
        </div>
        {isAdmin ? (
          <label className="text-xs text-gray-600 flex items-center gap-2">
            <input
              type="checkbox"
              checked={showDisabledSandboxProfiles}
              onChange={(e) => setShowDisabledSandboxProfiles(e.target.checked)}
            />
            Show disabled
          </label>
        ) : null}
      </div>
      <div className="space-y-2 max-h-64 overflow-auto">
        {filteredScientificSandboxProfiles.map((profile) => (
          <div key={String(profile.id)} className="border border-gray-200 rounded bg-white p-2">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <div className="font-medium text-gray-900">{String(profile.name)}</div>
                  <span className={`text-[11px] px-2 py-0.5 rounded ${profile.enabled ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'}`}>
                    {profile.enabled ? 'enabled' : 'disabled'}
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-indigo-100 text-indigo-700">
                    {String(profile.track_type || 'generic')}
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                    {profile.system_managed ? 'system' : 'custom'}
                  </span>
                  {profile.is_default ? (
                    <span className="text-[11px] px-2 py-0.5 rounded bg-amber-100 text-amber-800">default</span>
                  ) : null}
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  {String(profile.backend || 'docker')} · {String(profile.docker_image || 'no image')}
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  Timeout {Number(profile.timeout_seconds || 0)}s · Budget {Number(profile.budget_limit_default || 0)}
                </div>
              </div>
              {isAdmin ? (
                <div className="flex gap-1 shrink-0">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setEditingScientificSandboxProfileId(String(profile.id));
                      setSandboxProfileDraft(buildScientificSandboxProfileDraft(profile));
                    }}
                  >
                    Edit
                  </Button>
                  {!profile.system_managed ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={deleteScientificSandboxProfileMutation.isLoading}
                      onClick={() => deleteScientificSandboxProfileMutation.mutate(String(profile.id))}
                    >
                      Delete
                    </Button>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>
        ))}
        {filteredScientificSandboxProfiles.length === 0 ? (
          <div className="text-xs text-gray-500">No sandbox profiles available.</div>
        ) : null}
      </div>
      {isAdmin ? (
        <details className="border border-gray-200 rounded bg-white p-3" open={Boolean(editingScientificSandboxProfileId)}>
          <summary className="cursor-pointer text-xs font-medium text-gray-800">
            {editingScientificSandboxProfileId ? 'Edit sandbox profile' : 'Create custom sandbox profile'}
          </summary>
          <div className="mt-3 space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="Profile id"
                value={sandboxProfileDraft.id}
                disabled={Boolean(editingScientificSandboxProfileId)}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, id: e.target.value }))}
              />
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="Display name"
                value={sandboxProfileDraft.name}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, name: e.target.value }))}
              />
            </div>
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              rows={2}
              placeholder="Description"
              value={sandboxProfileDraft.description}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, description: e.target.value }))}
            />
            <div className="grid grid-cols-2 gap-2">
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                value={sandboxProfileDraft.track_type}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, track_type: e.target.value }))}
              >
                {DOMAIN_TRACK_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="Docker image"
                value={sandboxProfileDraft.docker_image}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, docker_image: e.target.value }))}
              />
            </div>
            <div className="grid grid-cols-4 gap-2">
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="Timeout"
                value={sandboxProfileDraft.timeout_seconds}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, timeout_seconds: e.target.value }))}
              />
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="Memory MB"
                value={sandboxProfileDraft.memory_mb}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, memory_mb: e.target.value }))}
              />
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="CPUs"
                value={sandboxProfileDraft.cpus}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, cpus: e.target.value }))}
              />
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="PIDs"
                value={sandboxProfileDraft.pids_limit}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, pids_limit: e.target.value }))}
              />
            </div>
            <input
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              placeholder="Budget limit"
              value={sandboxProfileDraft.budget_limit_default}
              disabled={editingScientificSandboxSystemManaged}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, budget_limit_default: e.target.value }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              rows={2}
              placeholder="Benchmark families, one per line"
              value={sandboxProfileDraft.allowed_benchmark_families}
              disabled={editingScientificSandboxSystemManaged}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, allowed_benchmark_families: e.target.value }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              rows={2}
              placeholder="Perf collectors, one per line"
              value={sandboxProfileDraft.allowed_perf_collectors}
              disabled={editingScientificSandboxSystemManaged}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, allowed_perf_collectors: e.target.value }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              rows={2}
              placeholder="Required capabilities, one per line"
              value={sandboxProfileDraft.required_capabilities}
              disabled={editingScientificSandboxSystemManaged}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, required_capabilities: e.target.value }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              rows={2}
              placeholder="Toolchains, one per line"
              value={sandboxProfileDraft.toolchains}
              disabled={editingScientificSandboxSystemManaged}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, toolchains: e.target.value }))}
            />
            <div className="flex items-center gap-4 text-xs text-gray-700">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={Boolean(sandboxProfileDraft.enabled)}
                  onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, enabled: e.target.checked }))}
                />
                Enabled
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={Boolean(sandboxProfileDraft.is_default)}
                  onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, is_default: e.target.checked }))}
                />
                Default for track
              </label>
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="primary"
                disabled={createScientificSandboxProfileMutation.isLoading || updateScientificSandboxProfileMutation.isLoading}
                onClick={submitScientificSandboxDraft}
              >
                {editingScientificSandboxProfileId ? 'Save Profile' : 'Create Profile'}
              </Button>
              <Button size="sm" variant="ghost" onClick={resetScientificSandboxDraft}>
                Reset
              </Button>
            </div>
          </div>
        </details>
      ) : null}
    </div>
  );

  return (
    <div className="p-6 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Autonomous Agents</h1>
          <p className="text-gray-500">Manage background jobs that work autonomously toward goals</p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowSystemMap((prev) => !prev)}>
            <MapIcon className="w-4 h-4 mr-2" />
            {showSystemMap ? 'Hide System Map' : 'System Map'}
          </Button>
          <Button variant="secondary" onClick={() => setShowCustomerResearchModal(true)}>
            <Brain className="w-4 h-4 mr-2" />
            Customer Research
          </Button>
          <Button onClick={() => setShowCreateModal(true)}>
            <Plus className="w-4 h-4 mr-2" />
            New Job
          </Button>
        </div>
      </div>

      {showSystemMap && (
        <div className="mb-6 rounded-lg border border-slate-200 bg-slate-50 p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">System Map</h2>
              <p className="mt-1 text-sm text-slate-600">
                Current operator surface and runtime ownership. Canonical doc: <code>docs/ARCHITECTURE_ASCII.md</code>
              </p>
            </div>
            <span className="rounded bg-white px-2 py-1 text-xs text-slate-500 border border-slate-200">
              Canonical autonomy: <code>automation_profile</code> / <code>automation_policy</code> / <code>effective_policy</code>
            </span>
          </div>
          <pre className="mt-4 overflow-x-auto rounded border border-slate-200 bg-white p-3 text-xs leading-5 text-slate-700">
            {AUTONOMOUS_SYSTEM_MAP}
          </pre>
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div className="mb-6">
          <div className="grid grid-cols-8 gap-4">
            <StatsCard title="Total Jobs" value={stats.total_jobs} icon={FileText} color="bg-gray-100 text-gray-600" />
            <StatsCard title="Running" value={stats.running_jobs} icon={Play} color="bg-blue-100 text-blue-600" />
            <StatsCard title="Completed" value={stats.completed_jobs} icon={CheckCircle2} color="bg-green-100 text-green-600" />
            <StatsCard title="Failed" value={stats.failed_jobs} icon={AlertCircle} color="bg-red-100 text-red-600" />
            <StatsCard
              title="Guard Blocks"
              value={jobCountSummary.scope_guard_blocked}
              icon={AlertCircle}
              color="bg-rose-100 text-rose-600"
              active={scopeGuardFilter === 'blocked'}
              onClick={() =>
                setScopeGuardFilter((prev) =>
                  prev === 'blocked' ? '' : 'blocked'
                )
              }
              titleHint="Toggle filter: scope guard blocked"
            />
            <StatsCard
              title="Bootstrap"
              value={jobCountSummary.bootstrap_recovered}
              icon={RefreshCw}
              color="bg-blue-100 text-blue-600"
              active={experimentRecoveryFilter === 'bootstrap_recovered'}
              onClick={() =>
                setExperimentRecoveryFilter((prev) =>
                  prev === 'bootstrap_recovered' ? '' : 'bootstrap_recovered'
                )
              }
              titleHint="Toggle filter: bootstrap recovered"
            />
            <StatsCard
              title="Fallback"
              value={jobCountSummary.fallback_attempted}
              icon={RotateCcw}
              color="bg-indigo-100 text-indigo-600"
              active={experimentRecoveryFilter === 'fallback_attempted'}
              onClick={() =>
                setExperimentRecoveryFilter((prev) =>
                  prev === 'fallback_attempted' ? '' : 'fallback_attempted'
                )
              }
              titleHint="Toggle filter: fallback attempted"
            />
            <StatsCard
              title="Failed Cmds"
              value={jobCountSummary.failed_command_total}
              icon={AlertCircle}
              color="bg-amber-100 text-amber-700"
              titleHint="Total failed verification commands across visible jobs"
            />
            <StatsCard
              title="Open Failures"
              value={jobCountSummary.unresolved_failed_command_total}
              icon={XCircle}
              color="bg-rose-100 text-rose-700"
              titleHint="Failed verification commands on fallback-attempted jobs that did not end in fallback success"
            />
            <StatsCard
              title="Open Recovery Jobs"
              value={jobCountSummary.unresolved_recovery_jobs}
              icon={AlertCircle}
              color="bg-rose-100 text-rose-700"
              active={experimentRecoveryFilter === 'unresolved_recovery'}
              onClick={() =>
                setExperimentRecoveryFilter((prev) =>
                  prev === 'unresolved_recovery' ? '' : 'unresolved_recovery'
                )
              }
              titleHint="Jobs whose latest fallback attempt still did not end in fallback success"
            />
            <StatsCard
              title="Claude QS"
              value={Number((stats.launch_mode_counts || {}).quick_start_claude_backend || 0)}
              icon={Sparkles}
              color="bg-indigo-100 text-indigo-600"
              active={launchModeFilter === 'quick_start_claude_backend'}
              onClick={() =>
                setLaunchModeFilter((prev) =>
                  prev === 'quick_start_claude_backend' ? '' : 'quick_start_claude_backend'
                )
              }
              titleHint="Toggle filter: Quick Start Claude Backend"
            />
            <StatsCard
              title="Domain QS"
              value={Number((stats.launch_mode_counts || {}).quick_start_domain_research || 0)}
              icon={Brain}
              color="bg-cyan-100 text-cyan-600"
              active={launchModeFilter === 'quick_start_domain_research'}
              onClick={() =>
                setLaunchModeFilter((prev) =>
                  prev === 'quick_start_domain_research' ? '' : 'quick_start_domain_research'
                )
              }
              titleHint="Toggle filter: Quick Start Domain Research"
            />
            <StatsCard
              title="Bug Swarm QS"
              value={Number((stats.launch_mode_counts || {}).quick_start_bug_triage_swarm || 0)}
              icon={GitBranch}
              color="bg-rose-100 text-rose-600"
              active={launchModeFilter === 'quick_start_bug_triage_swarm'}
              onClick={() =>
                setLaunchModeFilter((prev) =>
                  prev === 'quick_start_bug_triage_swarm' ? '' : 'quick_start_bug_triage_swarm'
                )
              }
              titleHint="Toggle filter: Quick Start Bug Triage Swarm"
            />
            <StatsCard
              title="Build Swarm QS"
              value={Number((stats.launch_mode_counts || {}).quick_start_build_break_swarm || 0)}
              icon={Layers}
              color="bg-amber-100 text-amber-700"
              active={launchModeFilter === 'quick_start_build_break_swarm'}
              onClick={() =>
                setLaunchModeFilter((prev) =>
                  prev === 'quick_start_build_break_swarm' ? '' : 'quick_start_build_break_swarm'
                )
              }
              titleHint="Toggle filter: Quick Start Build Break Swarm"
            />
            <StatsCard
              title="Frontend Swarm QS"
              value={Number((stats.launch_mode_counts || {}).quick_start_frontend_regression_swarm || 0)}
              icon={Sparkles}
              color="bg-cyan-100 text-cyan-700"
              active={launchModeFilter === 'quick_start_frontend_regression_swarm'}
              onClick={() =>
                setLaunchModeFilter((prev) =>
                  prev === 'quick_start_frontend_regression_swarm' ? '' : 'quick_start_frontend_regression_swarm'
                )
              }
              titleHint="Toggle filter: Quick Start Frontend Regression Swarm"
            />
            <StatsCard
              title="Bug Triage QS"
              value={Number((stats.launch_mode_counts || {}).quick_start_repo_bug_triage || 0)}
              icon={Bug}
              color="bg-amber-100 text-amber-600"
              active={launchModeFilter === 'quick_start_repo_bug_triage'}
              onClick={() =>
                setLaunchModeFilter((prev) =>
                  prev === 'quick_start_repo_bug_triage' ? '' : 'quick_start_repo_bug_triage'
                )
              }
              titleHint="Toggle filter: Quick Start Repo Bug Triage"
            />
            <StatsCard
              title="Role QS"
              value={Number((stats.launch_mode_counts || {}).quick_start_role_workflow || 0)}
              icon={Layers}
              color="bg-teal-100 text-teal-600"
              active={launchModeFilter === 'quick_start_role_workflow'}
              onClick={() =>
                setLaunchModeFilter((prev) =>
                  prev === 'quick_start_role_workflow' ? '' : 'quick_start_role_workflow'
                )
              }
              titleHint="Toggle filter: Quick Start Role Workflow"
            />
            <StatsCard
              title="Success Rate"
              value={stats.success_rate ? `${(stats.success_rate * 100).toFixed(0)}%` : '-'}
              icon={BarChart3}
              color="bg-purple-100 text-purple-600"
            />
          </div>
          {((stats.launch_mode_counts && Object.keys(stats.launch_mode_counts).length > 0) || Number((stats as any).launch_mode_none_count || 0) > 0) && (
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-gray-500">Launch modes:</span>
              <button
                type="button"
                className={`px-2 py-1 rounded-full border ${
                  launchModeFilter === '__none__'
                    ? 'border-amber-300 bg-amber-100 text-amber-800'
                    : 'border-amber-100 bg-amber-50 text-amber-700'
                }`}
                onClick={() => setLaunchModeFilter((prev) => (prev === '__none__' ? '' : '__none__'))}
                title="Filter jobs with no launch mode (manual/legacy)"
              >
                no-launch {Number((stats as any).launch_mode_none_count || 0)}
              </button>
              {Object.entries(stats.launch_mode_counts || {})
                .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
                .slice(0, 6)
                .map(([mode, count]) => (
                  <button
                    key={mode}
                    type="button"
                    className={`px-2 py-1 rounded-full border ${
                      launchModeFilter === mode
                        ? 'border-indigo-300 bg-indigo-100 text-indigo-800'
                        : 'border-gray-200 bg-gray-50 text-gray-700'
                    }`}
                    onClick={() => setLaunchModeFilter((prev) => (prev === mode ? '' : mode))}
                    title={`Filter jobs by launch mode: ${mode}`}
                  >
                    {mode} {count}
                  </button>
                ))}
            </div>
          )}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-4 mb-4 border-b border-gray-200">
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'queue'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('queue')}
        >
          <AlertCircle className="w-4 h-4" />
          Checkpoint Queue
          {checkpointQueueData?.total ? (
            <span className="ml-1 text-xs bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded">
              {checkpointQueueData.total}
            </span>
          ) : null}
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'trace'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('trace')}
        >
          <Clock className="w-4 h-4" />
          Decision Trace
          {decisionTraceData?.total ? (
            <span className="ml-1 text-xs bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded">
              {decisionTraceData.total}
            </span>
          ) : null}
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium ${
            activeTab === 'health'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('health')}
        >
          Autonomy Health
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium ${
            activeTab === 'jobs'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('jobs')}
        >
          My Jobs
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'swarm'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('swarm')}
        >
          <GitBranch className="w-4 h-4" />
          Swarm Review
          {swarmReviewJobs.length > 0 ? (
            <span className="ml-1 text-xs bg-rose-100 text-rose-700 px-1.5 py-0.5 rounded">
              {swarmReviewJobs.length}
            </span>
          ) : null}
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'outcomes'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('outcomes')}
        >
          <BarChart3 className="w-4 h-4" />
          Swarm Outcomes
          {Number((swarmOutcomeAnalyticsData as any)?.totals?.verified_fix_runs || 0) > 0 ? (
            <span className="ml-1 text-xs bg-emerald-100 text-emerald-700 px-1.5 py-0.5 rounded">
              {Number((swarmOutcomeAnalyticsData as any)?.totals?.verified_fix_runs || 0)}
            </span>
          ) : null}
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'profiles'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('profiles')}
        >
          <Settings className="w-4 h-4" />
          Swarm Profiles
          {codingSwarmProfiles.length > 0 ? (
            <span className="ml-1 text-xs bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded">
              {codingSwarmProfiles.length}
            </span>
          ) : null}
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium ${
            activeTab === 'backlog'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('backlog')}
        >
          Coding Backlog
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'domain'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('domain')}
        >
          <Brain className="w-4 h-4" />
          Domain Profiles
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'fleet'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('fleet')}
        >
          <Sparkles className="w-4 h-4" />
          Research Fleet
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'inbox'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('inbox')}
        >
          <Inbox className="w-4 h-4" />
          Research Inbox
          {inboxStats?.new ? (
            <span className="ml-1 text-xs bg-primary-100 text-primary-700 px-1.5 py-0.5 rounded">
              {inboxStats.new}
            </span>
          ) : null}
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium ${
            activeTab === 'templates'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('templates')}
        >
          Templates
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'chains'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('chains')}
        >
          <GitBranch className="w-4 h-4" />
          Job Chains
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 flex gap-6 min-h-0">
        {activeTab === 'queue' && (
          <div className="w-full flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3 text-sm text-gray-600">
                <span className="font-medium text-gray-900">Operator Queue</span>
                <span className="bg-amber-100 text-amber-700 px-2 py-1 rounded">Approvals: {checkpointQueueData?.approvals ?? '-'}</span>
                <span className="bg-red-100 text-red-700 px-2 py-1 rounded">Recoveries: {checkpointQueueData?.recoveries ?? '-'}</span>
                <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded">Follow-ups: {checkpointQueueData?.follow_ups ?? '-'}</span>
                <span className="bg-rose-100 text-rose-700 px-2 py-1 rounded">Policy reviews: {checkpointQueueData?.policy_reviews ?? '-'}</span>
                <span className="bg-amber-100 text-amber-700 px-2 py-1 rounded">Budget reviews: {checkpointQueueData?.budget_reviews ?? '-'}</span>
                <span className="bg-rose-100 text-rose-700 px-2 py-1 rounded">Overdue: {checkpointQueueData?.by_sla_bucket?.overdue || 0}</span>
                <span className="bg-amber-100 text-amber-800 px-2 py-1 rounded">At risk: {checkpointQueueData?.by_sla_bucket?.at_risk || 0}</span>
              </div>
              <div className="flex gap-2">
                <Button variant="secondary" onClick={() => setShowInboxMonitorModal(true)}>
                  <Activity className="w-4 h-4 mr-2" />
                  Create Monitor
                </Button>
                <Button variant="ghost" onClick={() => refetchCheckpointQueue()}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <div className="flex flex-wrap gap-2 mb-4">
              {[
                { value: '', label: 'All', count: checkpointQueueData?.total || 0 },
                { value: 'approval_checkpoint', label: 'Approvals', count: checkpointQueueData?.by_type?.approval_checkpoint || 0 },
                { value: 'job_recovery', label: 'Recoveries', count: checkpointQueueData?.by_type?.job_recovery || 0 },
                { value: 'follow_up_recommendation', label: 'Follow-ups', count: checkpointQueueData?.by_type?.follow_up_recommendation || 0 },
                { value: 'policy_review', label: 'Policy Reviews', count: checkpointQueueData?.by_type?.policy_review || 0 },
                { value: 'budget_review', label: 'Budget Reviews', count: checkpointQueueData?.by_type?.budget_review || 0 },
                { value: 'overdue', label: 'Overdue', count: checkpointQueueData?.by_sla_bucket?.overdue || 0, mode: 'sla' },
                { value: 'at_risk', label: 'At Risk', count: checkpointQueueData?.by_sla_bucket?.at_risk || 0, mode: 'sla' },
              ].map((chip) => (
                <button
                  key={`${chip.mode || 'type'}-${chip.value || 'all'}`}
                  type="button"
                  className={`px-3 py-1.5 rounded-full border text-sm ${
                    (chip.mode === 'sla' ? queueSlaBucketFilter === chip.value : queueItemTypeFilter === chip.value)
                      ? 'border-primary-300 bg-primary-100 text-primary-800'
                      : 'border-gray-200 bg-gray-50 text-gray-700'
                  }`}
                  onClick={() => {
                    if (chip.mode === 'sla') setQueueSlaBucketFilter((prev) => (prev === chip.value ? '' : chip.value));
                    else setQueueItemTypeFilter(chip.value);
                  }}
                >
                  {chip.label} {chip.count}
                </button>
              ))}
            </div>

            <div className="flex gap-3 mb-4 flex-wrap">
              {[
                { value: '', label: 'All queue work' },
                { value: 'compiler', label: 'Compiler only' },
                { value: 'approval_required', label: 'Approval-required follow-ups' },
                { value: 'blocked_validation', label: 'Blocked validations' },
                { value: 'failed_follow_up', label: 'Failed follow-ups' },
              ].map((preset) => (
                <button
                  key={`queue-preset-${preset.value || 'all'}`}
                  type="button"
                  className={`px-3 py-2 rounded-full border text-sm ${
                    queueOperatorPreset === preset.value
                      ? 'border-sky-300 bg-sky-50 text-sky-800'
                      : 'border-gray-200 bg-white text-gray-700'
                  }`}
                  onClick={() => setQueueOperatorPreset(preset.value)}
                >
                  {preset.label}
                </button>
              ))}
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={queueStatusFilter}
                onChange={(e) => setQueueStatusFilter(e.target.value)}
              >
                <option value="">Any status</option>
                {Object.entries(checkpointQueueData?.by_status || {}).map(([value, count]) => (
                  <option key={value} value={value}>
                    {value} ({count})
                  </option>
                ))}
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={queueCustomerFilter}
                onChange={(e) => setQueueCustomerFilter(e.target.value)}
              >
                <option value="">Any customer</option>
                {queueCustomerOptions.map(([value, count]) => (
                  <option key={value} value={value === 'Unassigned' ? '' : value}>
                    {value} ({count})
                  </option>
                ))}
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={queueJobTypeFilter}
                onChange={(e) => setQueueJobTypeFilter(e.target.value)}
              >
                <option value="">Any job type</option>
                <option value="research">Research</option>
                <option value="monitor">Monitor</option>
                <option value="analysis">Analysis</option>
                <option value="synthesis">Synthesis</option>
                <option value="custom">Custom</option>
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={queueSlaBucketFilter}
                onChange={(e) => setQueueSlaBucketFilter(e.target.value)}
              >
                <option value="">Any SLA</option>
                {Object.entries(checkpointQueueData?.by_sla_bucket || {}).map(([value, count]) => (
                  <option key={value} value={value}>
                    {value} ({count})
                  </option>
                ))}
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={queueEscalationFilter}
                onChange={(e) => setQueueEscalationFilter(e.target.value)}
              >
                <option value="">Any escalation</option>
                {Object.entries(checkpointQueueData?.by_escalation_level || {}).map(([value, count]) => (
                  <option key={value} value={value}>
                    {value} ({count})
                  </option>
                ))}
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={queueSortBy}
                onChange={(e) => setQueueSortBy(e.target.value)}
              >
                <option value="priority_score_desc">Urgency</option>
                <option value="sla_desc">SLA / escalation</option>
                <option value="age_desc">Oldest first</option>
                <option value="priority_desc">Base priority</option>
                <option value="created_desc">Newest first</option>
                <option value="created_asc">Oldest first</option>
              </select>
              <label className="inline-flex items-center gap-2 text-sm text-gray-700 px-2">
                <input
                  type="checkbox"
                  className="rounded border-gray-300"
                  checked={queueOverdueOnly}
                  onChange={(e) => setQueueOverdueOnly(Boolean(e.target.checked))}
                />
                Overdue only
              </label>
              {(queueItemTypeFilter || queueStatusFilter || queueCustomerFilter || queueJobTypeFilter || queueSlaBucketFilter || queueEscalationFilter || queueOverdueOnly || queueSortBy !== 'priority_score_desc' || queueOperatorPreset) && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setQueueItemTypeFilter('');
                    setQueueStatusFilter('');
                    setQueueCustomerFilter('');
                    setQueueJobFilter('');
                    setQueueHealthDrilldown('');
                    setQueueJobTypeFilter('');
                    setQueueSlaBucketFilter('');
                    setQueueEscalationFilter('');
                    setQueueOverdueOnly(false);
                    setQueueSortBy('priority_score_desc');
                    setQueueOperatorPreset('');
                  }}
                >
                  <XCircle className="w-4 h-4 mr-1" />
                  Clear Filters
                </Button>
              )}
            </div>

            {queueHealthDrilldown ? (
              <div className="flex items-center gap-2 mb-4 text-xs">
                <span className="bg-sky-50 text-sky-800 border border-sky-200 px-2 py-1 rounded">
                  Showing follow-up recommendations
                  {queueCustomerFilter ? ` for ${queueCustomerFilter}` : ''}
                  {queueJobFilter ? ` · ${queueJobFilter}` : ''}
                  {` · ${formatQueueHealthDrilldownLabel(queueHealthDrilldown)}`}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setQueueHealthDrilldown('');
                    navigate(buildAutonomousAgentsUrl(undefined, {
                      queue_health_drilldown: null,
                    }), { replace: true });
                  }}
                >
                  Clear drilldown
                </Button>
              </div>
            ) : null}

            {checkpointQueueLoading ? (
              <div className="flex justify-center items-center flex-1">
                <LoadingSpinner />
              </div>
            ) : visibleQueueItems.length === 0 ? (
              <div className="flex flex-col items-center justify-center flex-1 text-gray-500">
                <CheckCircle2 className="w-12 h-12 mb-3 text-gray-400" />
                <p className="text-lg font-medium">Queue is clear</p>
                <p className="text-sm">Approvals, recurring job recoveries, and accepted-signal follow-ups will appear here.</p>
              </div>
            ) : (
              <div className="space-y-3 overflow-y-auto flex-1 pr-1">
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                  <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div className="flex flex-wrap items-center gap-2">
                      <Button size="sm" variant="ghost" onClick={selectVisibleQueueItems}>
                        Select Visible
                      </Button>
                      <Button size="sm" variant="ghost" onClick={clearQueueSelection}>
                        Clear Selection
                      </Button>
                      <span className="text-xs text-slate-600">
                        Selected {selectedQueueItems.length} of {visibleQueueItems.length}
                      </span>
                    </div>
                    {queueBulkState.itemType ? (
                      <div className="flex flex-wrap items-center gap-2">
                        {queueBulkState.itemType === 'approval_checkpoint' || queueBulkState.itemType === 'follow_up_recommendation' ? (
                          <input
                            className="border border-gray-300 rounded-lg px-3 py-2 text-sm min-w-[220px]"
                            placeholder={
                              queueBulkState.itemType === 'follow_up_recommendation'
                                ? 'Shared note for selected follow-ups'
                                : 'Shared note for selected approvals'
                            }
                            value={queueBulkNote}
                            onChange={(e) => setQueueBulkNote(e.target.value)}
                          />
                        ) : null}
                        {queueBulkState.actions.map((action) => (
                          <Button
                            key={`bulk-${action}`}
                            size="sm"
                            variant={action === 'reject' || action === 'cancel' || action === 'reject_launch' ? 'ghost' : 'primary'}
                            disabled={bulkQueueActionMutation.isLoading || bulkQueueFollowUpActionMutation.isLoading}
                            onClick={() => {
                              if (queueBulkState.itemType === 'follow_up_recommendation') {
                                const firstItem = selectedQueueItems[0];
                                const domainProfileId = String(firstItem?.domain_research_profile_id || '').trim();
                                const portfolioId = String(firstItem?.portfolio_id || '').trim();
                                bulkQueueFollowUpActionMutation.mutate({
                                  domain_research_profile_id: domainProfileId || undefined,
                                  profile_opportunity_ids: domainProfileId
                                    ? selectedQueueItems.map((item) => String(item.profile_opportunity_id || '').trim()).filter(Boolean)
                                    : undefined,
                                  portfolio_id: portfolioId || undefined,
                                  portfolio_opportunity_ids: portfolioId
                                    ? selectedQueueItems.map((item) => String(item.portfolio_opportunity_id || '').trim()).filter(Boolean)
                                    : undefined,
                                  action: action as 'approve_launch' | 'reject_launch',
                                  operator_note: queueBulkNote.trim() || undefined,
                                });
                                return;
                              }
                              bulkQueueActionMutation.mutate({
                                itemType: queueBulkState.itemType as 'approval_checkpoint' | 'job_recovery',
                                action: action as 'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel',
                                jobIds: selectedQueueItems
                                  .map((item) => String(item.job_id || ''))
                                  .filter(Boolean),
                                checkpointNote: queueBulkState.itemType === 'approval_checkpoint'
                                  ? (queueBulkNote.trim() || undefined)
                                  : undefined,
                              });
                            }}
                          >
                            {action === 'approve_launch'
                              ? 'Approve selected'
                              : action === 'reject_launch'
                                ? 'Reject selected'
                                : action.replace(/_/g, ' ')}
                          </Button>
                        ))}
                      </div>
                    ) : (
                      <div className="text-xs text-slate-600">
                        {queueBulkState.disabledReason}
                      </div>
                    )}
                  </div>
                </div>
                {visibleQueueItems.map((item: AgentCheckpointQueueItem) => (
                  <div key={item.queue_key} className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="pt-1">
                        <input
                          type="checkbox"
                          className="rounded border-gray-300"
                          checked={!!queueSelection[item.queue_key]}
                          onChange={() => toggleQueueSelection(item)}
                          aria-label={`Select queue item ${item.title}`}
                        />
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`text-xs px-2 py-1 rounded ${
                            item.item_type === 'approval_checkpoint'
                              ? 'bg-amber-100 text-amber-800'
                              : item.item_type === 'job_recovery'
                                ? 'bg-red-100 text-red-800'
                                : item.item_type === 'policy_review'
                                  ? 'bg-rose-100 text-rose-800'
                                  : item.item_type === 'budget_review'
                                    ? 'bg-amber-100 text-amber-800'
                                  : 'bg-blue-100 text-blue-800'
                          }`}>
                            {item.item_type.replace(/_/g, ' ')}
                          </span>
                          {item.status ? (
                            <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
                              {item.status}
                            </span>
                          ) : null}
                          {item.reason_label ? (
                            <span className="text-xs bg-slate-100 text-slate-700 px-2 py-1 rounded">
                              {item.reason_label}
                            </span>
                          ) : null}
                          {item.sla_bucket ? (
                            <span className={`text-xs px-2 py-1 rounded ${
                              item.sla_bucket === 'overdue'
                                ? 'bg-rose-100 text-rose-800'
                                : item.sla_bucket === 'at_risk'
                                  ? 'bg-amber-100 text-amber-800'
                                  : 'bg-emerald-100 text-emerald-800'
                            }`}>
                              {item.sla_bucket.replace(/_/g, ' ')}
                            </span>
                          ) : null}
                          {item.escalation_level ? (
                            <span className={`text-xs px-2 py-1 rounded ${
                              item.escalation_level === 'high'
                                ? 'bg-rose-50 text-rose-700 border border-rose-200'
                                : item.escalation_level === 'medium'
                                  ? 'bg-amber-50 text-amber-700 border border-amber-200'
                                  : 'bg-slate-50 text-slate-600 border border-slate-200'
                            }`}>
                              {item.escalation_level}
                            </span>
                          ) : null}
                          {item.created_at ? (
                            <span className="text-xs text-gray-500">
                              {new Date(item.created_at).toLocaleString()}
                            </span>
                          ) : null}
                        </div>
                        <div className="font-medium text-gray-900">{item.title}</div>
                        {item.summary ? (
                          <div className="text-sm text-gray-600 mt-1">{item.summary}</div>
                        ) : null}
                        {item.evidence_summary ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Evidence: {item.evidence_summary}
                          </div>
                        ) : null}
                        <div className="text-xs text-gray-500 mt-2 flex flex-wrap gap-3">
                          {typeof item.age_minutes === 'number' ? <span>Age: {item.age_minutes}m</span> : null}
                          {typeof item.priority_score === 'number' ? <span>Urgency: {item.priority_score}</span> : null}
                          {item.is_overdue ? <span className="text-rose-700 font-medium">Overdue</span> : null}
                          {item.is_stale ? <span className="text-rose-700 font-medium">Stale</span> : null}
                        </div>
                        {item.checkpoint?.action?.tool ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Pending tool: <span className="font-mono">{String(item.checkpoint.action.tool)}</span>
                          </div>
                        ) : null}
                        {item.customer ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Customer: {item.customer}
                          </div>
                        ) : null}
                        {item.portfolio_title ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Fleet: {item.portfolio_title}
                            {item.portfolio_opportunity_key ? <span> · {item.portfolio_opportunity_key}</span> : null}
                          </div>
                        ) : null}
                        {item.domain_research_profile_title ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Domain profile: {item.domain_research_profile_title}
                            {item.profile_opportunity_key ? <span> · {item.profile_opportunity_key}</span> : null}
                          </div>
                        ) : null}
                        {item.job_type ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Job type: {item.job_type}
                          </div>
                        ) : null}
                        {(item.domain || item.objective || item.track_type || item.source_scope) ? (
                          <div className="mt-2 rounded-lg border border-sky-100 bg-sky-50 p-2 text-xs text-sky-900 space-y-1">
                            {item.domain ? <div>Domain: {item.domain}</div> : null}
                            {item.objective ? <div>Objective: {item.objective}</div> : null}
                            {item.track_type ? <div>Track: {item.track_type.replace(/_/g, ' ')}</div> : null}
                            {item.source_scope ? <div>Source scope: {item.source_scope.replace(/_/g, ' ')}</div> : null}
                            {item.repo_source_ids?.length ? <div>Repo inputs: {item.repo_source_ids.slice(0, 3).join(', ')}</div> : null}
                            {item.benchmark_queries?.length ? <div>Benchmarks: {item.benchmark_queries.slice(0, 2).join(' · ')}</div> : null}
                            {item.sandbox_profile_id ? <div>Sandbox: {item.sandbox_profile_id}</div> : null}
                            {item.automation_profile ? <div>Automation profile: {item.automation_profile.replace(/_/g, ' ')}</div> : null}
                            {item.effective_policy?.follow_up_review_mode ? (
                              <div>Review mode: {String(item.effective_policy.follow_up_review_mode).replace(/_/g, ' ')}</div>
                            ) : null}
                            {typeof item.confidence === 'number' ? <div>Confidence: {(Number(item.confidence) * 100).toFixed(0)}%</div> : null}
                            {typeof item.readiness === 'number' ? <div>Readiness: {(Number(item.readiness) * 100).toFixed(0)}%</div> : null}
                          </div>
                        ) : null}
                        {(item.linked_note_ids?.length || item.linked_experiment_plan_ids?.length || item.linked_validation_run_ids?.length || item.child_job_ids?.length) ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Links:
                            {item.linked_note_ids?.length ? <span> notes {item.linked_note_ids.length}</span> : null}
                            {item.linked_experiment_plan_ids?.length ? <span> · plans {item.linked_experiment_plan_ids.length}</span> : null}
                            {item.linked_validation_run_ids?.length ? <span> · validations {item.linked_validation_run_ids.length}</span> : null}
                            {item.child_job_ids?.length ? <span> · child jobs {item.child_job_ids.length}</span> : null}
                          </div>
                        ) : null}
                        {item.recommended_action ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Recommended: {item.recommended_action}
                          </div>
                        ) : null}
                        {item.item_type === 'follow_up_recommendation' && item.actions?.find((row) => row.recommended)?.recommendation_score !== undefined ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Follow-up score: {item.actions?.find((row) => row.recommended)?.recommendation_score}
                            {item.actions?.find((row) => row.recommended)?.recommendation_reasons?.length ? (
                              <span> · why: {item.actions?.find((row) => row.recommended)?.recommendation_reasons?.slice(0, 3).join(', ')}</span>
                            ) : null}
                          </div>
                        ) : null}
                        {item.follow_up_policy_mode ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Follow-up policy: {item.follow_up_policy_mode.replace(/_/g, ' ')}
                          </div>
                        ) : null}
                        {item.follow_up_launch_status ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Follow-up status: {item.follow_up_launch_status.replace(/_/g, ' ')}
                          </div>
                        ) : null}
                        {item.follow_up_block_reason ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Follow-up note: {item.follow_up_block_reason}
                          </div>
                        ) : null}
                        {item.follow_up_budget_decision || item.budget_throttle_state ? (
                          <div className="text-xs text-amber-700 mt-2">
                            Budget: {(item.follow_up_budget_decision || item.budget_throttle_state || '').replace(/_/g, ' ')}
                            {item.follow_up_budget_reason || item.budget_reason ? (
                              <span> · {item.follow_up_budget_reason || item.budget_reason}</span>
                            ) : null}
                          </div>
                        ) : null}
                        {item.follow_up_customer_budget_decision || item.customer_budget_throttle_state ? (
                          <div className="text-xs text-rose-700 mt-2">
                            Customer budget: {(item.follow_up_customer_budget_decision || item.customer_budget_throttle_state || '').replace(/_/g, ' ')}
                            {item.follow_up_customer_budget_reason || item.customer_budget_reason ? (
                              <span> · {item.follow_up_customer_budget_reason || item.customer_budget_reason}</span>
                            ) : null}
                          </div>
                        ) : null}
                        {item.policy_guardrail_action ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Safeguard: {item.policy_guardrail_action.replace(/_/g, ' ')}
                            {item.policy_guardrail_target_policy?.follow_up_review_mode || item.policy_guardrail_follow_up_autonomy?.mode ? (
                              <span>
                                {' '}to {String(item.policy_guardrail_target_policy?.follow_up_review_mode || item.policy_guardrail_follow_up_autonomy?.mode).replace(/_/g, ' ')}
                              </span>
                            ) : null}
                          </div>
                        ) : null}
                        {(item.policy_guardrail_reasons || []).length ? (
                          <div className="text-xs text-rose-700 mt-2">
                            {(item.policy_guardrail_reasons || []).slice(0, 2).join(' · ')}
                          </div>
                        ) : null}
                        {item.follow_up_operator_decision ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Operator decision: {item.follow_up_operator_decision.replace(/_/g, ' ')}
                          </div>
                        ) : null}
                        {item.follow_up_operator_acted_at ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Acted at {new Date(String(item.follow_up_operator_acted_at)).toLocaleString()}
                          </div>
                        ) : null}
                        {item.item_type === 'follow_up_recommendation' && item.follow_up_launch_status === 'pending_approval' ? (
                          <div className="mt-3">
                            <textarea
                              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                              rows={2}
                              placeholder="Operator note (optional)"
                              value={getQueueDraftValue(item).note}
                              onChange={(e) => setQueueDraftValue(item, { note: e.target.value })}
                            />
                          </div>
                        ) : null}
                        {item.follow_up_job_id ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Follow-up job: {item.follow_up_job_id}
                          </div>
                        ) : null}
                        {item.next_run_at ? (
                          <div className="text-xs text-gray-500 mt-2">
                            Next run at {new Date(String(item.next_run_at)).toLocaleString()}
                          </div>
                        ) : null}
                        {(() => {
                          const schedulerState = (item.job?.scheduler_state && typeof item.job.scheduler_state === 'object')
                            ? item.job.scheduler_state
                            : item.scheduler_state;
                          const summary = summarizeSchedulerState(schedulerState);
                          if (summary.length === 0) return null;
                          return (
                            <div className="mt-2 rounded-lg border border-gray-200 bg-gray-50 p-2 text-xs text-gray-600 space-y-1">
                              {summary.slice(0, 4).map((line) => (
                                <div key={line}>{line}</div>
                              ))}
                            </div>
                          );
                        })()}
                        {item.item_type === 'approval_checkpoint' && (() => {
                          const draft = getQueueDraftValue(item);
                          return (
                            <div className="mt-3 space-y-2">
                              <textarea
                                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                                rows={2}
                                placeholder="Operator note (optional)"
                                value={draft.note}
                                onChange={(e) => setQueueDraftValue(item, { note: e.target.value })}
                              />
                              <div className="flex items-center gap-2">
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => setQueueDraftValue(item, { showEdit: !draft.showEdit })}
                                >
                                  {draft.showEdit ? 'Hide Edit' : 'Edit Action'}
                                </Button>
                              </div>
                              {draft.showEdit && (
                                <div className="grid grid-cols-1 gap-2 border border-gray-200 rounded-lg p-3 bg-gray-50">
                                  <input
                                    className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                                    placeholder="Tool name"
                                    value={draft.tool}
                                    onChange={(e) => setQueueDraftValue(item, { tool: e.target.value })}
                                  />
                                  <input
                                    className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                                    placeholder="Purpose"
                                    value={draft.purpose}
                                    onChange={(e) => setQueueDraftValue(item, { purpose: e.target.value })}
                                  />
                                  <textarea
                                    className="border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                                    rows={5}
                                    placeholder='{"source_id": "..."}'
                                    value={draft.params}
                                    onChange={(e) => setQueueDraftValue(item, { params: e.target.value })}
                                  />
                                </div>
                              )}
                            </div>
                          );
                        })()}
                      </div>
                      <div className="flex flex-col items-end gap-2 shrink-0">
                        {(item.domain_research_profile_id || item.portfolio_id) ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => openQueueItemTarget(item)}
                          >
                            {item.domain_research_profile_id ? 'Open Domain' : 'Open Fleet'}
                          </Button>
                        ) : null}
                        {item.job_id ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => openQueueItemTarget(item)}
                          >
                            View Job
                          </Button>
                        ) : null}
                        {item.actions?.map((action) => (
                          <Button
                            key={`${item.queue_key}-${action.label}`}
                            size="sm"
                            variant={action.kind === 'job_action' ? 'secondary' : 'primary'}
                            disabled={
                              actionMutation.isLoading ||
                              followUpQueueActionMutation.isLoading ||
                              bulkQueueActionMutation.isLoading ||
                              bulkQueueFollowUpActionMutation.isLoading ||
                              createMutation.isLoading ||
                              createFromChainMutation.isLoading
                            }
                            onClick={() => {
                              if (action.kind === 'job_action' && item.job_id && action.action) {
                                runQueueAction(item, action.action as any);
                                return;
                              }
                              if (action.kind === 'policy_action' && action.action) {
                                runQueuePolicyAction(item, action);
                                return;
                              }
                              if (action.kind === 'follow_up_action' && action.action) {
                                const payload = (action.follow_up_action_payload || {}) as Record<string, any>;
                                followUpQueueActionMutation.mutate({
                                  inbox_item_id: payload.inbox_item_id ? String(payload.inbox_item_id) : (item.inbox_item_id ? String(item.inbox_item_id) : undefined),
                                  domain_research_profile_id: payload.domain_research_profile_id ? String(payload.domain_research_profile_id) : (item.domain_research_profile_id ? String(item.domain_research_profile_id) : undefined),
                                  profile_opportunity_id: payload.profile_opportunity_id ? String(payload.profile_opportunity_id) : (item.profile_opportunity_id ? String(item.profile_opportunity_id) : undefined),
                                  portfolio_id: payload.portfolio_id ? String(payload.portfolio_id) : (item.portfolio_id ? String(item.portfolio_id) : undefined),
                                  portfolio_opportunity_id: payload.portfolio_opportunity_id ? String(payload.portfolio_opportunity_id) : (item.portfolio_opportunity_id ? String(item.portfolio_opportunity_id) : undefined),
                                  action: action.action as 'approve_launch' | 'reject_launch',
                                  operator_note: getQueueDraftValue(item).note.trim() || undefined,
                                });
                                return;
                              }
                              if (action.chain_create_payload) {
                                launchQueueRecommendation(item, action.chain_create_payload as Record<string, any>);
                                return;
                              }
                              if (action.job_create_payload) {
                                launchQueueRecommendation(item, action.job_create_payload as Record<string, any>);
                              }
                            }}
                            title={action.description || undefined}
                          >
                            {action.recommended ? `${action.label}` : action.label}
                          </Button>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'trace' && (
          <div className="w-full flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Decision Trace</h2>
                <p className="text-sm text-gray-500">
                  Canonical operator-facing event feed across queue, monitors, domain profiles, fleets, jobs, and validation runs.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" onClick={() => downloadDecisionTraceExport('json')}>
                  <Download className="w-4 h-4 mr-1" />
                  JSON
                </Button>
                <Button variant="ghost" size="sm" onClick={() => downloadDecisionTraceExport('csv')}>
                  <Download className="w-4 h-4 mr-1" />
                  CSV
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    refetchDecisionTrace();
                    refetchDecisionTraceAnalytics();
                  }}
                >
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-3 mb-4">
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Unassigned New</div>
                <div className="mt-2 text-2xl font-semibold text-rose-700">
                  {(decisionTraceData?.items || []).filter((event) => String(event.triage_status || '').trim().toLowerCase() === 'new' && !event.assigned_to_user_id).length}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Assigned Investigating</div>
                <div className="mt-2 text-2xl font-semibold text-blue-700">
                  {(decisionTraceData?.items || []).filter((event) => String(event.triage_status || '').trim().toLowerCase() === 'investigating' && !!event.assigned_to_user_id).length}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Escalated</div>
                <div className="mt-2 text-2xl font-semibold text-amber-700">{Number(decisionTraceData?.by_escalation_state?.escalated || 0)}</div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Overdue</div>
                <div className="mt-2 text-2xl font-semibold text-fuchsia-700">{Number(decisionTraceData?.overdue_count || 0)}</div>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-4 gap-3 mb-4">
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500 flex items-center justify-between gap-2">
                  <span>Trace mix</span>
                  <span className="text-gray-400">{Number(decisionTraceAnalyticsData?.total || 0)} rows</span>
                </div>
                <div className="mt-2 space-y-2">
                  <div className="text-[11px] font-medium text-gray-500">Decision types</div>
                  <div className="flex flex-wrap gap-2">
                    {(summarizeTraceAnalyticsBuckets(decisionTraceAnalyticsData?.top_decision_types) || []).length ? (
                      summarizeTraceAnalyticsBuckets(decisionTraceAnalyticsData?.top_decision_types).map((value) => (
                        <span key={value} className="inline-flex items-center rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-700">
                          {value}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-gray-400">{decisionTraceAnalyticsLoading ? 'Loading analytics...' : 'No trace analytics yet'}</span>
                    )}
                  </div>
                  <div className="border-t border-gray-100 pt-2 text-[11px] text-gray-500">
                    <div className="flex flex-wrap gap-2">
                      {(summarizeTraceAnalyticsBuckets(
                        Object.entries(decisionTraceAnalyticsData?.by_source_kind || {}).map(([value, count]) => ({ value, count }))
                      ) || []).slice(0, 3).map((value) => (
                        <span key={value} className="inline-flex items-center rounded-full bg-emerald-50 px-2 py-1 text-[11px] text-emerald-700">
                          Source {value}
                        </span>
                      ))}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {(summarizeTraceAnalyticsBuckets(
                        Object.entries(decisionTraceAnalyticsData?.by_triage_status || {}).map(([value, count]) => ({ value, count }))
                      ) || []).slice(0, 3).map((value) => (
                        <span key={value} className="inline-flex items-center rounded-full bg-blue-50 px-2 py-1 text-[11px] text-blue-700">
                          Triage {value}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Reason labels</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {summarizeTraceAnalyticsBuckets(decisionTraceAnalyticsData?.top_reason_labels).length ? (
                    summarizeTraceAnalyticsBuckets(decisionTraceAnalyticsData?.top_reason_labels).map((value) => (
                      <span key={value} className="inline-flex items-center rounded-full bg-fuchsia-100 px-2 py-1 text-xs text-fuchsia-700">
                        {value}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-gray-400">{decisionTraceAnalyticsLoading ? 'Loading analytics...' : 'No trace analytics yet'}</span>
                  )}
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Queue reasons</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {summarizeTraceAnalyticsBuckets(decisionTraceAnalyticsData?.top_queue_reasons).length ? (
                    summarizeTraceAnalyticsBuckets(decisionTraceAnalyticsData?.top_queue_reasons).map((value) => (
                      <span key={value} className="inline-flex items-center rounded-full bg-amber-100 px-2 py-1 text-xs text-amber-800">
                        {value}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-gray-400">{decisionTraceAnalyticsLoading ? 'Loading analytics...' : 'No queue reasons yet'}</span>
                  )}
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500 flex items-center justify-between gap-2">
                  <span>7-day trend</span>
                  <span className="text-gray-400">Last {decisionTraceAnalyticsData?.window_days || 7} days</span>
                </div>
                <div className="mt-2 grid grid-cols-7 gap-1 items-end">
                  {(decisionTraceAnalyticsData?.daily_trend || []).map((point) => {
                    const maxCount = Math.max(1, ...(decisionTraceAnalyticsData?.daily_trend || []).map((trend) => Number(trend.count || 0)));
                    const barHeight = Math.max(12, Math.round((Number(point.count || 0) / maxCount) * 72));
                    return (
                      <div key={point.day} className="flex flex-col items-center gap-1">
                        <div className="w-full h-20 flex items-end justify-center bg-gray-50 rounded-lg border border-gray-100">
                          <div
                            className="w-4 rounded-t bg-fuchsia-500"
                            style={{ height: `${barHeight}px` }}
                          />
                        </div>
                        <div className="text-[10px] text-gray-500 text-center leading-tight">
                          <div>{formatTraceAnalyticsDay(point.day)}</div>
                          <div className="font-medium text-gray-700">{Number(point.count || 0)}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-xl p-4 mb-4">
              <div className="flex items-center gap-3 flex-wrap">
                <select
                  aria-label="Trace saved view"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={selectedTraceViewId}
                  onChange={(e) => {
                    const viewId = e.target.value;
                    const nextView = (traceViewsData?.items || []).find((item) => item.id === viewId);
                    if (nextView) {
                      applyTraceView(nextView);
                    } else {
                      setSelectedTraceViewId('');
                      setTraceViewNameDraft('');
                      setTraceViewIsDefaultDraft(false);
                    }
                  }}
                >
                  <option value="">Saved views</option>
                  {(traceViewsData?.items || []).map((view) => (
                    <option key={view.id} value={view.id}>
                      {view.name}{view.is_default ? ' (Default)' : ''}
                    </option>
                  ))}
                </select>
                <input
                  aria-label="Trace view name"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm min-w-[220px]"
                  placeholder="Current view name"
                  value={traceViewNameDraft}
                  onChange={(e) => setTraceViewNameDraft(e.target.value)}
                />
                <label className="inline-flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-700">
                  <input
                    aria-label="Default trace view"
                    type="checkbox"
                    className="h-4 w-4 rounded border-gray-300 text-fuchsia-600 focus:ring-fuchsia-500"
                    checked={traceViewIsDefaultDraft}
                    onChange={(e) => setTraceViewIsDefaultDraft(e.target.checked)}
                  />
                  <span>Default view</span>
                </label>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    const name = String(traceViewNameDraft || '').trim();
                    if (!name) {
                      toast.error('Name the trace view first');
                      return;
                    }
                    createTraceViewMutation.mutate({
                      name,
                      filters: currentTraceViewFilters,
                      is_default: traceViewIsDefaultDraft,
                    });
                  }}
                >
                  Save Current View
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={async () => {
                    const link = `${window.location.origin}${buildTraceShareUrl(location.search)}`;
                    try {
                      if (navigator?.clipboard?.writeText) {
                        await navigator.clipboard.writeText(link);
                        toast.success('Trace link copied');
                      } else {
                        toast.error('Clipboard copy is not available in this browser');
                      }
                    } catch {
                      toast.error('Failed to copy trace link');
                    }
                  }}
                  title="Copy a shareable deep link for the current trace filters"
                >
                  <Link2 className="w-4 h-4 mr-1" />
                  Copy Trace Link
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={!selectedTraceViewId}
                  onClick={() => {
                    if (!selectedTraceViewId) return;
                    updateTraceViewMutation.mutate({
                      viewId: selectedTraceViewId,
                      payload: {
                        name: String(traceViewNameDraft || '').trim() || undefined,
                        filters: currentTraceViewFilters,
                        is_default: traceViewIsDefaultDraft,
                      },
                    });
                  }}
                >
                  Update View
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={!selectedTraceViewId}
                  onClick={() => {
                    if (!selectedTraceViewId) return;
                    deleteTraceViewMutation.mutate(selectedTraceViewId);
                  }}
                >
                  Delete View
                </Button>
              </div>
            </div>

            <div className="flex gap-3 mb-4 flex-wrap">
              {[
                { value: '', label: 'All trace events' },
                { value: 'compiler', label: 'Compiler only' },
                { value: 'approval_required', label: 'Approval-required follow-ups' },
                { value: 'blocked_validation', label: 'Blocked validations' },
                { value: 'failed_follow_up', label: 'Failed follow-ups' },
                { value: 'reevaluation_closeout', label: 'Reevaluation closeouts' },
              ].map((preset) => (
                <button
                  key={`trace-preset-${preset.value || 'all'}`}
                  type="button"
                  className={`px-3 py-2 rounded-full border text-sm ${
                    traceOperatorPreset === preset.value
                      ? 'border-sky-300 bg-sky-50 text-sky-800'
                      : 'border-gray-200 bg-white text-gray-700'
                  }`}
                  onClick={() => setTraceOperatorPreset(preset.value)}
                >
                  {preset.label}
                </button>
              ))}
            </div>

            <div className="flex gap-3 mb-4 flex-wrap">
                <select
                  aria-label="Trace source filter"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={traceSourceKindFilter}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceSourceKindFilter(e.target.value);
                  }}
                >
                <option value="">All sources</option>
                {Object.entries(decisionTraceData?.by_source_kind || {}).map(([value, count]) => (
                  <option key={value} value={value}>
                    {humanizeDecisionTraceValue(value)} ({count})
                  </option>
                ))}
              </select>
                <select
                  aria-label="Trace date range filter"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={traceDateRange}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceDateRange(e.target.value);
                  }}
                >
                <option value="24h">Last 24h</option>
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
                <option value="all">All time</option>
              </select>
                <select
                  aria-label="Trace decision filter"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={traceDecisionTypeFilter}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceDecisionTypeFilter(e.target.value);
                  }}
                >
                <option value="">All decisions</option>
                {Object.entries(decisionTraceData?.by_decision_type || {}).map(([value, count]) => (
                  <option key={value} value={value}>
                    {humanizeDecisionTraceValue(value)} ({count})
                  </option>
                ))}
              </select>
                <select
                  aria-label="Trace status filter"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={traceStatusFilter}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceStatusFilter(e.target.value);
                  }}
                >
                <option value="">Any status</option>
                {Object.entries(decisionTraceData?.by_status || {}).map(([value, count]) => (
                  <option key={value} value={value === 'unknown' ? '' : value}>
                    {humanizeDecisionTraceValue(value)} ({count})
                  </option>
                ))}
              </select>
                <select
                  aria-label="Trace severity filter"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={traceSeverityFilter}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceSeverityFilter(e.target.value);
                  }}
                >
                <option value="">Any severity</option>
                {Object.entries(decisionTraceData?.by_severity || {}).map(([value, count]) => (
                  <option key={value} value={value === 'unknown' ? '' : value}>
                    {humanizeDecisionTraceValue(value)} ({count})
                  </option>
                ))}
              </select>
                <select
                  aria-label="Trace actor filter"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={traceActorModeFilter}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceActorModeFilter(e.target.value);
                  }}
                >
                <option value="">All actors</option>
                {Object.entries(decisionTraceData?.by_actor_mode || {}).map(([value, count]) => (
                  <option key={value} value={value === 'unknown' ? '' : value}>
                    {humanizeDecisionTraceValue(value)} ({count})
                  </option>
                ))}
              </select>
                <select
                  aria-label="Trace triage filter"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={traceTriageStatusFilter}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceTriageStatusFilter(e.target.value);
                  }}
                >
                <option value="">Any triage state</option>
                {Object.entries(decisionTraceData?.by_triage_status || {}).map(([value, count]) => (
                  <option key={value} value={value === 'unknown' ? '' : value}>
                    {humanizeDecisionTraceValue(value)} ({count})
                  </option>
                ))}
              </select>
                <select
                  aria-label="Trace assignee filter"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={traceAssignedToUserIdFilter}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceAssignedToUserIdFilter(e.target.value);
                  }}
                >
                <option value="">Any assignee</option>
                {collaborationUsers.map((candidate) => (
                  <option key={candidate.id} value={String(candidate.id)}>
                    {userLabelById(String(candidate.id))}
                  </option>
                ))}
              </select>
                <select
                  aria-label="Trace escalation filter"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={traceEscalationStateFilter}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceEscalationStateFilter(e.target.value);
                  }}
                >
                <option value="">Any escalation</option>
                {Object.entries(decisionTraceData?.by_escalation_state || {}).map(([value, count]) => (
                  <option key={value} value={value === 'none' ? '' : value}>
                    {humanizeDecisionTraceValue(value)} ({count})
                  </option>
                ))}
              </select>
                <select
                  aria-label="Trace customer filter"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={traceCustomerFilter}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceCustomerFilter(e.target.value);
                  }}
                >
                <option value="">All customers</option>
                {Object.entries(decisionTraceData?.by_customer || {}).map(([value, count]) => (
                  <option key={value} value={value === 'Unassigned' ? '' : value}>
                    {value} ({count})
                  </option>
                ))}
              </select>
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={traceUnassignedOnly}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceUnassignedOnly(e.target.checked);
                  }}
                />
                Unassigned only
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={tracePinnedOnly}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTracePinnedOnly(e.target.checked);
                  }}
                />
                Pinned only
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={traceActionableOnly}
                  onChange={(e) => {
                    markTraceFiltersDirty();
                    setTraceActionableOnly(e.target.checked);
                  }}
                />
                Actionable only
              </label>
            </div>

            <div className="bg-white border border-gray-200 rounded-xl overflow-hidden flex-1 min-h-0">
              <div className="grid grid-cols-[190px_190px_1fr_220px] gap-4 px-4 py-3 text-xs font-semibold uppercase tracking-wide text-gray-500 border-b border-gray-200 bg-gray-50">
                <span>Time</span>
                <span>Source</span>
                <span>Decision</span>
                <span>State</span>
              </div>
              <div className="divide-y divide-gray-100 overflow-y-auto h-full">
                {decisionTraceLoading ? (
                  <div className="px-4 py-8 text-sm text-gray-500">Loading decision trace...</div>
                ) : !visibleDecisionTraceItems.length ? (
                  <div className="px-4 py-8 text-sm text-gray-500">No decision trace events match the current filters.</div>
                ) : (
                  visibleDecisionTraceItems.map((event) => {
                    const isExpanded = expandedTraceEventId === event.event_id;
                    const metadata = (event.metadata && typeof event.metadata === 'object')
                      ? (event.metadata as Record<string, any>)
                      : {};
                    const ownerLabel = event.owner_label || userLabelById(String(event.owner_user_id || '')) || String(event.owner_user_id || '').slice(0, 8);
                    const assigneeLabel = event.assignee_label || userLabelById(String(event.assigned_to_user_id || '')) || String(event.assigned_to_user_id || '').slice(0, 8);
                    const assigneeDraftValue = traceAssigneeDrafts[event.event_id] ?? String(event.assigned_to_user_id || '');
                    const dueAtDraftValue = traceDueAtDrafts[event.event_id] ?? toDecisionTraceDueInputValue(event.due_at);
                    const actionNoteDraftValue = traceActionNoteDrafts[event.event_id] ?? String(event.resolution_note || event.operator_note || '');
                    const supportsFollowUpApproval = canActOnTraceFollowUpEvent(event);
                    const supportsFollowUpRelaunch = canRelaunchTraceFollowUpEvent(event);
                    const schedulerState = (event.scheduler_state && typeof event.scheduler_state === 'object')
                      ? (event.scheduler_state as Record<string, any>)
                      : (metadata.scheduler_state && typeof metadata.scheduler_state === 'object')
                        ? (metadata.scheduler_state as Record<string, any>)
                        : null;
                    const schedulerSummary = summarizeSchedulerState(schedulerState);
                    const sourceNoteId = String(metadata.source_note_id || '').trim();
                    const targetNoteId = String(metadata.target_note_id || '').trim();
                    const reevaluationJobId = String(metadata.reevaluation_job_id || metadata.review_job_id || '').trim();
                    const isReevaluationCloseout = String(event.event_type || event.decision_type || '').trim().toLowerCase().startsWith('reevaluation_');
                    const afterStateText = Object.entries(event.after_state || {})
                      .filter(([, value]) => value !== null && value !== undefined && String(value).trim() !== '')
                      .slice(0, 2)
                      .map(([key, value]) => `${humanizeDecisionTraceValue(key)}: ${typeof value === 'string' ? humanizeDecisionTraceValue(value) : JSON.stringify(value)}`)
                      .join(' • ');
                    return (
                      <div key={event.event_id}>
                        <button
                          type="button"
                          className="grid grid-cols-[190px_190px_1fr_220px] gap-4 px-4 py-4 text-sm w-full text-left hover:bg-gray-50"
                          onClick={() => setExpandedTraceEventId((current) => (current === event.event_id ? '' : event.event_id))}
                        >
                          <div className="text-gray-600">
                            <div>{new Date(event.event_time).toLocaleString()}</div>
                            <div className="text-xs text-gray-400 mt-1">{humanizeDecisionTraceValue(event.event_type)}</div>
                          </div>
                          <div>
                            <div className="font-medium text-gray-900">{event.source_label || humanizeDecisionTraceValue(event.source_kind)}</div>
                            <div className="text-xs text-gray-500 mt-1">{humanizeDecisionTraceValue(event.source_kind)}</div>
                            {event.customer ? <div className="text-xs text-gray-500">{event.customer}</div> : null}
                          </div>
                          <div>
                            <div className="font-medium text-gray-900">{event.summary}</div>
                            <div className="text-xs text-gray-500 mt-1">
                              {humanizeDecisionTraceValue(event.decision_type)}
                              {event.reason_code ? ` • ${humanizeDecisionTraceValue(event.reason_code)}` : ''}
                              {event.reason_label ? ` • ${humanizeDecisionTraceValue(event.reason_label)}` : ''}
                            </div>
                            <div className="mt-2 flex items-center gap-2 flex-wrap">
                              <span className={`px-2 py-1 rounded-full text-xs font-medium ${event.is_derived ? 'bg-amber-100 text-amber-700' : decisionTraceTriageClasses(event.triage_status)}`}>
                                {event.is_derived ? 'Read only' : humanizeDecisionTraceValue(event.triage_status || 'new')}
                              </span>
                              {event.escalation_state && event.escalation_state !== 'none' ? (
                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${event.escalation_state === 'escalated' ? 'bg-amber-100 text-amber-800' : 'bg-yellow-100 text-yellow-800'}`}>
                                  {humanizeDecisionTraceValue(event.escalation_state)}
                                </span>
                              ) : null}
                              {event.pinned ? (
                                <span className="px-2 py-1 rounded-full text-xs font-medium bg-indigo-100 text-indigo-700">
                                  Pinned
                                </span>
                              ) : null}
                            </div>
                            {event.is_derived ? (
                              <div className="mt-2 inline-flex px-2 py-1 rounded-full bg-amber-50 text-amber-700 text-xs font-medium">
                                Derived fallback
                              </div>
                            ) : null}
                            {event.operator_note ? (
                              <div className="text-xs text-gray-600 mt-2">Note: {event.operator_note}</div>
                            ) : null}
                            {event.reason_label ? (
                              <div className="text-xs text-gray-600 mt-2">
                                Reason label: {humanizeDecisionTraceValue(event.reason_label)}
                              </div>
                            ) : null}
                            {(event.domain || event.objective || event.track_type || event.source_scope) ? (
                              <div className="mt-2 rounded-lg border border-sky-100 bg-sky-50 p-2 text-xs text-sky-900 space-y-1">
                                {event.domain ? <div>Domain: {event.domain}</div> : null}
                                {event.objective ? <div>Objective: {event.objective}</div> : null}
                                {event.track_type ? <div>Track: {event.track_type.replace(/_/g, ' ')}</div> : null}
                                {event.source_scope ? <div>Source scope: {event.source_scope.replace(/_/g, ' ')}</div> : null}
                                {event.repo_source_ids?.length ? <div>Repo inputs: {event.repo_source_ids.slice(0, 3).join(', ')}</div> : null}
                                {event.benchmark_queries?.length ? <div>Benchmarks: {event.benchmark_queries.slice(0, 2).join(' · ')}</div> : null}
                                {event.sandbox_profile_id ? <div>Sandbox: {event.sandbox_profile_id}</div> : null}
                                {event.automation_profile ? <div>Automation profile: {event.automation_profile.replace(/_/g, ' ')}</div> : null}
                                {event.effective_policy?.follow_up_review_mode ? (
                                  <div>Review mode: {String(event.effective_policy.follow_up_review_mode).replace(/_/g, ' ')}</div>
                                ) : null}
                                {typeof event.confidence === 'number' ? <div>Confidence: {(Number(event.confidence) * 100).toFixed(0)}%</div> : null}
                                {typeof event.readiness === 'number' ? <div>Readiness: {(Number(event.readiness) * 100).toFixed(0)}%</div> : null}
                              </div>
                            ) : null}
                            {(event.linked_note_ids?.length || event.linked_experiment_plan_ids?.length || event.linked_validation_run_ids?.length || event.child_job_ids?.length) ? (
                              <div className="text-xs text-gray-600 mt-2">
                                Links:
                                {event.linked_note_ids?.length ? <span> notes {event.linked_note_ids.length}</span> : null}
                                {event.linked_experiment_plan_ids?.length ? <span> · plans {event.linked_experiment_plan_ids.length}</span> : null}
                                {event.linked_validation_run_ids?.length ? <span> · validations {event.linked_validation_run_ids.length}</span> : null}
                                {event.child_job_ids?.length ? <span> · child jobs {event.child_job_ids.length}</span> : null}
                              </div>
                            ) : null}
                            {schedulerSummary.length ? (
                              <div className="mt-2 rounded-lg border border-gray-200 bg-white p-2 text-xs text-gray-600 space-y-1">
                                {schedulerSummary.slice(0, 4).map((line) => (
                                  <div key={line}>{line}</div>
                                ))}
                              </div>
                            ) : null}
                            {event.owner_user_id ? (
                              <div className="text-xs text-gray-600 mt-2">
                                Owner: {ownerLabel || String(event.owner_user_id).slice(0, 8)}
                                {event.is_owned_by_current_user ? ' · Me' : ''}
                              </div>
                            ) : null}
                            {event.assigned_to_user_id ? (
                              <div className="text-xs text-gray-600 mt-2">
                                Assignee: {assigneeLabel || String(event.assigned_to_user_id).slice(0, 8)}
                                {event.is_assigned_to_current_user ? ' · Me' : ''}
                              </div>
                            ) : null}
                            {event.due_at ? (
                              <div className="text-xs text-gray-600 mt-1">Due: {new Date(event.due_at).toLocaleString()}</div>
                            ) : null}
                            {event.deep_link ? (
                              <span className="mt-2 inline-flex text-xs font-medium text-primary-600">
                                {event.deep_link.label || 'Open Source'}
                              </span>
                            ) : null}
                          </div>
                          <div className="flex flex-col gap-2 items-start">
                            {event.status ? (
                              <span className={`px-2 py-1 rounded-full text-xs font-medium ${decisionTraceSeverityClasses(event.severity)}`}>
                                {humanizeDecisionTraceValue(event.status)}
                              </span>
                            ) : null}
                            {event.severity ? (
                              <span className="text-xs text-gray-500">Severity: {humanizeDecisionTraceValue(event.severity)}</span>
                            ) : null}
                            {event.actor_mode ? (
                              <span className="text-xs text-gray-500">Actor: {humanizeDecisionTraceValue(event.actor_mode)}</span>
                            ) : null}
                            {event.team_bucket ? (
                              <span className="text-xs text-gray-500">Team: {humanizeDecisionTraceValue(event.team_bucket)}</span>
                            ) : null}
                            {event.escalation_reason ? (
                              <span className="text-xs text-gray-500">Escalation: {humanizeDecisionTraceValue(event.escalation_reason)}</span>
                            ) : null}
                            {afterStateText ? (
                              <div className="text-xs text-gray-500">{afterStateText}</div>
                            ) : null}
                          </div>
                        </button>
                        {isExpanded ? (
                          <div className="px-4 pb-4 pt-1 border-t border-gray-100 bg-gray-50">
                            <div className="grid grid-cols-2 gap-4 text-xs">
                              <div>
                                <div className="font-semibold text-gray-700 mb-2">Before</div>
                                <pre className="bg-white border border-gray-200 rounded-lg p-3 overflow-x-auto text-[11px] text-gray-700 whitespace-pre-wrap">
                                  {JSON.stringify(event.before_state || {}, null, 2)}
                                </pre>
                              </div>
                              <div>
                                <div className="font-semibold text-gray-700 mb-2">After</div>
                                <pre className="bg-white border border-gray-200 rounded-lg p-3 overflow-x-auto text-[11px] text-gray-700 whitespace-pre-wrap">
                                  {JSON.stringify(event.after_state || {}, null, 2)}
                                </pre>
                              </div>
                            </div>
                            {event.metadata ? (
                              <div className="mt-4">
                                <div className="font-semibold text-gray-700 mb-2 text-xs">Metadata</div>
                                <pre className="bg-white border border-gray-200 rounded-lg p-3 overflow-x-auto text-[11px] text-gray-700 whitespace-pre-wrap">
                                  {JSON.stringify(event.metadata, null, 2)}
                                </pre>
                              </div>
                            ) : null}
                            {schedulerSummary.length ? (
                              <div className="mt-4">
                                <div className="font-semibold text-gray-700 mb-2 text-xs">Scheduler</div>
                                <div className="rounded-lg border border-gray-200 bg-white p-3 text-xs text-gray-600 space-y-1">
                                  {schedulerSummary.map((line) => (
                                    <div key={line}>{line}</div>
                                  ))}
                                </div>
                              </div>
                            ) : null}
                            <div className="mt-4 flex flex-wrap gap-2">
                              <Button
                                size="sm"
                                variant="secondary"
                                onClick={async () => {
                                  const link = `${window.location.origin}${buildTraceShareUrl(location.search, event.event_id)}`;
                                  try {
                                    if (navigator?.clipboard?.writeText) {
                                      await navigator.clipboard.writeText(link);
                                      toast.success('Event link copied');
                                    } else {
                                      toast.error('Clipboard copy is not available in this browser');
                                    }
                                  } catch {
                                    toast.error('Failed to copy event link');
                                  }
                                }}
                                title="Copy a permalink for this trace event and the current filters"
                              >
                                <Link2 className="w-4 h-4 mr-1" />
                                Copy Event Link
                              </Button>
                              {event.deep_link ? (
                                <Button size="sm" variant="secondary" onClick={() => openDecisionTraceTarget(event)}>
                                  {event.deep_link.label || 'Open Source'}
                                </Button>
                              ) : null}
                              {isReevaluationCloseout && reevaluationJobId ? (
                                <Button size="sm" variant="ghost" onClick={() => openDecisionTraceReevaluationJob(reevaluationJobId)}>
                                  Open reevaluation job
                                </Button>
                              ) : null}
                              {isReevaluationCloseout && sourceNoteId ? (
                                <Button size="sm" variant="ghost" onClick={() => openDecisionTraceResearchNote(sourceNoteId)}>
                                  Open source note
                                </Button>
                              ) : null}
                              {isReevaluationCloseout && targetNoteId && targetNoteId !== sourceNoteId ? (
                                <Button size="sm" variant="ghost" onClick={() => openDecisionTraceResearchNote(targetNoteId)}>
                                  Open saved note
                                </Button>
                              ) : null}
                            </div>
                            {!event.is_derived ? (
                              <div className="mt-4 space-y-3">
                                <div className="flex items-center gap-2 flex-wrap">
                                  <Button size="sm" variant="ghost" onClick={() => runDecisionTraceAction(event, 'acknowledge')}>
                                    Acknowledge
                                  </Button>
                                  <Button size="sm" variant="ghost" onClick={() => runDecisionTraceAction(event, 'start_investigation')}>
                                    Investigate
                                  </Button>
                                  <Button size="sm" variant="ghost" onClick={() => runDecisionTraceAction(event, 'unassign')}>
                                    Unassign
                                  </Button>
                                  <Button size="sm" variant="ghost" onClick={() => runDecisionTraceAction(event, 'clear_due_at')}>
                                    Clear Due
                                  </Button>
                                  <Button size="sm" variant="ghost" onClick={() => runDecisionTraceAction(event, 'toggle_pin')}>
                                    {event.pinned ? 'Unpin' : 'Pin'}
                                  </Button>
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)_auto] gap-2 items-end">
                                  <label className="block">
                                    <span className="block text-[11px] font-medium text-gray-600 mb-1">Assignee</span>
                                    <select
                                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white"
                                      value={assigneeDraftValue}
                                      onChange={(e) => setTraceAssigneeDrafts((current) => ({ ...current, [event.event_id]: e.target.value }))}
                                    >
                                      <option value="">Unassigned</option>
                                      {collaborationUsers.map((candidate) => (
                                        <option key={`trace-assignee-${event.event_id}-${candidate.id}`} value={String(candidate.id)}>
                                          {userLabelById(String(candidate.id)) || candidate.username || String(candidate.id)}
                                        </option>
                                      ))}
                                    </select>
                                  </label>
                                  <Button size="sm" variant="secondary" onClick={() => runDecisionTraceAssignmentAction(event)}>
                                    Apply Assignee
                                  </Button>
                                  <label className="block">
                                    <span className="block text-[11px] font-medium text-gray-600 mb-1">Due at</span>
                                    <input
                                      type="datetime-local"
                                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white"
                                      value={dueAtDraftValue}
                                      onChange={(e) => setTraceDueAtDrafts((current) => ({ ...current, [event.event_id]: e.target.value }))}
                                    />
                                  </label>
                                  <Button size="sm" variant="secondary" onClick={() => runDecisionTraceDueAtAction(event)}>
                                    Apply Due
                                  </Button>
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_auto] gap-2 items-end">
                                  <label className="block">
                                    <span className="block text-[11px] font-medium text-gray-600 mb-1">{supportsFollowUpApproval || supportsFollowUpRelaunch ? 'Operator note' : 'Action note'}</span>
                                    <textarea
                                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white"
                                      rows={2}
                                      placeholder={supportsFollowUpApproval ? 'Approval or rejection note' : supportsFollowUpRelaunch ? 'Relaunch note' : 'Resolution or reopen note'}
                                      value={actionNoteDraftValue}
                                      onChange={(e) => setTraceActionNoteDrafts((current) => ({ ...current, [event.event_id]: e.target.value }))}
                                    />
                                  </label>
                                  <div className="flex flex-wrap gap-2">
                                    {supportsFollowUpApproval ? (
                                      <>
                                        <Button size="sm" variant="primary" onClick={() => runDecisionTraceAction(event, 'approve_launch', actionNoteDraftValue)}>
                                          Approve
                                        </Button>
                                        <Button size="sm" variant="ghost" onClick={() => runDecisionTraceAction(event, 'reject_launch', actionNoteDraftValue)}>
                                          Reject
                                        </Button>
                                      </>
                                    ) : null}
                                    {supportsFollowUpRelaunch ? (
                                      <Button size="sm" variant="primary" onClick={() => runDecisionTraceAction(event, 'relaunch_follow_up', actionNoteDraftValue)}>
                                        Relaunch Follow-up
                                      </Button>
                                    ) : null}
                                    <Button size="sm" variant="ghost" onClick={() => runDecisionTraceAction(event, 'resolve', actionNoteDraftValue)}>
                                      Resolve
                                    </Button>
                                    <Button size="sm" variant="ghost" onClick={() => runDecisionTraceAction(event, 'reopen', actionNoteDraftValue)}>
                                      Reopen
                                    </Button>
                                  </div>
                                </div>
                              </div>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
            <div className="flex items-center justify-between mt-4 text-sm text-gray-600">
              <div>
                Showing {decisionTraceData?.items?.length || 0} of {decisionTraceData?.total || 0} events
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={traceOffset <= 0}
                  onClick={() => setTraceOffset((current) => Math.max(0, current - 50))}
                >
                  Previous
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={!decisionTraceData?.has_more}
                  onClick={() => setTraceOffset((current) => current + 50)}
                >
                  Next
                </Button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'health' && (
          <div className="w-full flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Autonomy Health</h2>
                <p className="text-sm text-gray-500">
                  Discovery quality and safe follow-up effectiveness across recurring monitors.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="secondary" onClick={() => setShowMonitorProfilesModal(true)}>
                  <Brain className="w-4 h-4 mr-1" />
                  Monitor Profiles
                </Button>
                <Button variant="ghost" size="sm" onClick={() => refetchMonitorAnalytics()}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-5 gap-3 mb-4">
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Monitors</div>
                <div className="mt-2 text-2xl font-semibold text-gray-900">{filteredMonitorAnalytics.totals.total_monitors}</div>
                <div className="text-xs text-gray-500 mt-1">
                  Strong {filteredMonitorAnalytics.totals.strong_monitors} · Mixed {filteredMonitorAnalytics.totals.mixed_monitors} · Weak {filteredMonitorAnalytics.totals.weak_monitors}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Discoveries</div>
                <div className="mt-2 text-2xl font-semibold text-gray-900">{filteredMonitorAnalytics.totals.discovered_count}</div>
                <div className="text-xs text-gray-500 mt-1">
                  Accepted {filteredMonitorAnalytics.totals.accepted_count} · Rejected {filteredMonitorAnalytics.totals.rejected_count}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Safe launches</div>
                <div className="mt-2 text-2xl font-semibold text-gray-900">
                  {filteredMonitorAnalytics.totals.auto_launched_count + filteredMonitorAnalytics.totals.approval_launched_count}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Auto {filteredMonitorAnalytics.totals.auto_launched_count} · Approved {filteredMonitorAnalytics.totals.approval_launched_count}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Outcomes</div>
                <div className="mt-2 text-2xl font-semibold text-gray-900">{filteredMonitorAnalytics.totals.follow_up_completed_count}</div>
                <div className="text-xs text-gray-500 mt-1">
                  Completed · Failed {filteredMonitorAnalytics.totals.follow_up_failed_count} · Cancelled {filteredMonitorAnalytics.totals.follow_up_cancelled_count}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Blocked</div>
                <div className="mt-2 text-2xl font-semibold text-gray-900">{filteredMonitorAnalytics.totals.blocked_count}</div>
                <div className="text-xs text-gray-500 mt-1">Accepted items that stayed manual or policy-blocked</div>
              </div>
            </div>

            {customerPortfolioRows.length > 0 ? (
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <h3 className="section-heading">Customer Fleet Health</h3>
                    <p className="text-xs text-gray-500">Cross-monitor autonomy load, backlog, alert pressure, and throttle state by customer.</p>
                  </div>
                </div>
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                  {customerPortfolioRows.map((customerRow) => {
                    const active = healthCustomerFilter === customerRow.customer;
                    const status = String(customerRow.portfolio_status || 'normal');
                    const customerBudgetDraft = getHealthCustomerBudgetDraft(customerRow);
                    const rebalancePreview = getHealthCustomerRebalancePreview(customerRow.customer);
                    const rebalanceUpdates = buildCustomerRebalanceUpdates(customerRow);
                    const customerBudgetChanged =
                      customerBudgetDraft.auto_launch_limit_24h !== Number(customerRow.customer_budget?.auto_launch_limit_24h || 0)
                      || customerBudgetDraft.approval_queue_limit_24h !== Number(customerRow.customer_budget?.approval_queue_limit_24h || 0)
                      || customerBudgetDraft.alert_limit_24h !== Number(customerRow.customer_budget?.alert_limit_24h || 0)
                      || customerBudgetDraft.queue_backlog_cap !== Number(customerRow.customer_budget?.queue_backlog_cap || 0);
                    return (
                      <div
                        key={customerRow.customer}
                        className={`border rounded-lg p-4 ${active ? 'border-primary-300 bg-primary-50' : 'border-gray-200 bg-white'}`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="flex items-center gap-2 flex-wrap">
                              <h4 className="font-medium text-gray-900">{customerRow.customer}</h4>
                              <span className={`text-xs px-2 py-1 rounded ${
                                status === 'monitor_throttled'
                                  || status === 'customer_budget_throttled'
                                  ? 'bg-rose-100 text-rose-700'
                                  : status === 'backlog_heavy' || status === 'alert_heavy' || status === 'nearing_saturation'
                                    ? 'bg-amber-100 text-amber-800'
                                    : 'bg-emerald-100 text-emerald-700'
                              }`}>
                                {status.replace(/_/g, ' ')}
                              </span>
                              <span className="text-xs bg-slate-100 text-slate-700 px-2 py-1 rounded">
                                {customerRow.monitor_count} monitor{customerRow.monitor_count === 1 ? '' : 's'}
                              </span>
                            </div>
                            <div className="mt-2 flex flex-wrap gap-2 text-[11px]">
                              <span className={`px-2 py-1 rounded ${
                                String(customerRow.customer_budget_throttle_state || 'normal') === 'normal'
                                  ? 'bg-emerald-100 text-emerald-700'
                                  : String(customerRow.customer_budget_throttle_state || '') === 'auto_launch_throttled'
                                    ? 'bg-amber-100 text-amber-800'
                                    : 'bg-rose-100 text-rose-700'
                              }`}>
                                Shared budget {String(customerRow.customer_budget_throttle_state || 'normal').replace(/_/g, ' ')}
                              </span>
                              <span className="bg-slate-100 text-slate-700 px-2 py-1 rounded">
                                Auto {customerRow.auto_launch_used_24h}/{customerRow.auto_launch_capacity_24h}
                              </span>
                              <span className="bg-slate-100 text-slate-700 px-2 py-1 rounded">
                                Queue {customerRow.approval_queue_used_24h}/{customerRow.approval_queue_capacity_24h}
                              </span>
                              <span className="bg-slate-100 text-slate-700 px-2 py-1 rounded">
                                Alerts {customerRow.alert_used_24h}/{customerRow.alert_capacity_24h}
                              </span>
                              <span className="bg-slate-100 text-slate-700 px-2 py-1 rounded">
                                Backlog {customerRow.backlog_used}/{customerRow.backlog_capacity}
                              </span>
                              <span className="bg-slate-100 text-slate-700 px-2 py-1 rounded">
                                Throttled {customerRow.throttled_monitor_count}
                              </span>
                            </div>
                            {(customerRow.portfolio_reasons || []).length > 0 ? (
                              <div className="mt-2 flex flex-wrap gap-2">
                                {(customerRow.portfolio_reasons || []).map((reason) => (
                                  <span key={reason} className="text-[11px] bg-white border border-gray-200 text-gray-700 px-2 py-1 rounded">
                                    {reason}
                                  </span>
                                ))}
                              </div>
                            ) : null}
                            {(customerRow.customer_budget_throttle_reasons || []).length > 0 ? (
                              <div className="mt-2 flex flex-wrap gap-2">
                                {(customerRow.customer_budget_throttle_reasons || []).map((reason) => (
                                  <span key={`budget-${reason}`} className="text-[11px] bg-amber-50 border border-amber-200 text-amber-800 px-2 py-1 rounded">
                                    {reason}
                                  </span>
                                ))}
                              </div>
                            ) : null}
                            {customerRow.latest_rebalance_evaluation_status ? (
                              <div className="mt-2 flex flex-wrap gap-2">
                                <span className={`text-[11px] px-2 py-1 rounded ${
                                  customerRow.latest_rebalance_evaluation_status === 'improving'
                                    ? 'bg-emerald-100 text-emerald-700'
                                    : customerRow.latest_rebalance_evaluation_status === 'degrading'
                                      ? 'bg-rose-100 text-rose-700'
                                      : customerRow.latest_rebalance_evaluation_status === 'mixed'
                                        ? 'bg-amber-100 text-amber-800'
                                        : 'bg-slate-100 text-slate-700'
                                }`}>
                                  Rebalance {formatPolicyEvaluationStatus(customerRow.latest_rebalance_evaluation_status)}
                                </span>
                                <span className="text-[11px] bg-white border border-gray-200 text-gray-700 px-2 py-1 rounded">
                                  Sample {customerRow.latest_rebalance_evaluation_sample_count}/{customerRow.latest_rebalance_evaluation_target_count || customerRow.latest_rebalance_evaluation_sample_count}
                                </span>
                              </div>
                            ) : null}
                          </div>
                          <div className="flex flex-col gap-2 shrink-0">
                            <Button
                              size="sm"
                              variant={active ? 'secondary' : 'ghost'}
                              onClick={() => setHealthCustomerFilter((prev) => (prev === customerRow.customer ? '' : customerRow.customer))}
                            >
                              {active ? 'Clear Filter' : 'Filter Monitors'}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => {
                                setActiveTab('queue');
                                navigate(buildAutonomousAgentsUrl(undefined, {
                                  tab: 'queue',
                                  queue_item_type: null,
                                  queue_customer: customerRow.customer,
                                  queue_job: null,
                                  queue_health_drilldown: null,
                                }), { replace: true });
                              }}
                            >
                              View Queue
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => {
                                setActiveTab('inbox');
                                setInboxCustomerFilter(customerRow.customer);
                                setInboxStatusFilter('accepted');
                                setInboxTypeFilter('');
                                setInboxSearch('');
                                navigate(buildAutonomousAgentsUrl(undefined, {
                                  tab: 'inbox',
                                  inbox_customer: customerRow.customer,
                                  inbox_job: null,
                                  inbox_health_drilldown: null,
                                  inbox_policy_drilldown: null,
                                }), { replace: true });
                              }}
                            >
                              View Inbox
                            </Button>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3 text-xs">
                          <div className="bg-gray-50 rounded p-3">
                            <div className="uppercase tracking-wide text-gray-500">Monitor mix</div>
                            <div className="text-gray-700 mt-1">Strong {customerRow.strong_monitor_count} · Mixed {customerRow.mixed_monitor_count} · Weak {customerRow.weak_monitor_count}</div>
                          </div>
                          <div className="bg-gray-50 rounded p-3">
                            <div className="uppercase tracking-wide text-gray-500">Follow-ups</div>
                            <div className="text-gray-700 mt-1">Accepted {customerRow.accepted_count} · Blocked {customerRow.blocked_count}</div>
                            <div className="mt-1 flex flex-wrap gap-2">
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => openQueueHealthDrilldown('blocked_follow_up', { customer: customerRow.customer })}
                              >
                                Blocked {Number(customerRow.blocked_count || 0)}
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => openQueueHealthDrilldown('pending_follow_up_approvals', { customer: customerRow.customer })}
                              >
                                Queue {Number(customerRow.approval_queue_used_24h || 0)}
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => openQueueHealthDrilldown('manual_follow_up_recommendations', { customer: customerRow.customer })}
                              >
                                Manual {Number(customerRow.blocked_count || 0)}
                              </Button>
                            </div>
                          </div>
                          <div className="bg-gray-50 rounded p-3">
                            <div className="uppercase tracking-wide text-gray-500">Outcomes</div>
                            <div className="mt-1 flex flex-wrap gap-2">
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => openInboxHealthDrilldown('completed_follow_up', { customer: customerRow.customer })}
                              >
                                Completed {customerRow.follow_up_completed_count}
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => openInboxHealthDrilldown('failed_follow_up', { customer: customerRow.customer })}
                              >
                                Failed {customerRow.follow_up_failed_count}
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => openInboxHealthDrilldown('cancelled_follow_up', { customer: customerRow.customer })}
                              >
                                Cancelled {customerRow.follow_up_cancelled_count}
                              </Button>
                            </div>
                          </div>
                          <div className="bg-gray-50 rounded p-3">
                            <div className="uppercase tracking-wide text-gray-500">Top pressure</div>
                            {(() => {
                              const targetMonitor = customerRow.throttled_monitors?.[0] || customerRow.top_backlog_monitors?.[0] || customerRow.top_alert_monitors?.[0] || customerRow.top_launch_monitors?.[0];
                              if (!targetMonitor?.monitor_job_id || !targetMonitor?.monitor_name) {
                                return <div className="text-gray-700 mt-1">No pressure</div>;
                              }
                              return (
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  className="mt-1 px-0"
                                  onClick={() => openHealthMonitorFocus(customerRow.customer, String(targetMonitor.monitor_job_id))}
                                >
                                  {String(targetMonitor.monitor_name)}
                                </Button>
                              );
                            })()}
                            {(customerRow.throttled_monitors || []).length > 0 ? (
                              <div className="mt-2 flex flex-wrap gap-2">
                                {(customerRow.throttled_monitors || []).map((monitor) => (
                                  <Button
                                    key={`${customerRow.customer}-throttled-${monitor.monitor_job_id}`}
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => openHealthMonitorFocus(customerRow.customer, String(monitor.monitor_job_id))}
                                  >
                                    {monitor.monitor_name}
                                  </Button>
                                ))}
                              </div>
                            ) : null}
                          </div>
                        </div>
                        <div className="mt-3 border border-gray-200 rounded p-3 bg-slate-50">
                          <div className="flex items-center justify-between gap-2 mb-3">
                            <div>
                              <div className="text-xs font-medium text-slate-700">Shared customer budget</div>
                              <div className="text-[11px] text-slate-500">
                                Auto {customerRow.customer_budget_usage?.auto_launch_count_24h || 0}/{customerRow.customer_budget?.auto_launch_limit_24h || 0}
                                {' · '}
                                Queue {customerRow.customer_budget_usage?.approval_queue_count_24h || 0}/{customerRow.customer_budget?.approval_queue_limit_24h || 0}
                                {' · '}
                                Alerts {customerRow.customer_budget_usage?.alert_count_24h || 0}/{customerRow.customer_budget?.alert_limit_24h || 0}
                                {' · '}
                                Backlog {customerRow.customer_budget_usage?.queue_backlog_count || 0}/{customerRow.customer_budget?.queue_backlog_cap || 0}
                              </div>
                            </div>
                            <div className="flex gap-2">
                              <Button
                                size="sm"
                                variant="ghost"
                                disabled={updateCustomerBudgetMutation.isLoading}
                                onClick={() => {
                                  updateCustomerBudgetMutation.mutate({
                                    customer: customerRow.customer,
                                    data: { reset_to_default: true },
                                  });
                                }}
                              >
                                Reset shared caps
                              </Button>
                              <Button
                                size="sm"
                                disabled={!customerBudgetChanged || updateCustomerBudgetMutation.isLoading}
                                onClick={() => {
                                  updateCustomerBudgetMutation.mutate({
                                    customer: customerRow.customer,
                                    data: customerBudgetDraft,
                                  });
                                }}
                              >
                                Save shared caps
                              </Button>
                            </div>
                          </div>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {[
                              ['Auto launches', 'auto_launch_limit_24h'],
                              ['Approval queue', 'approval_queue_limit_24h'],
                              ['Alerts', 'alert_limit_24h'],
                              ['Backlog cap', 'queue_backlog_cap'],
                            ].map(([label, key]) => (
                              <label key={`${customerRow.customer}-${key}`} className="block">
                                <div className="text-[11px] text-slate-600 mb-1">{label}</div>
                                <input
                                  type="number"
                                  min={0}
                                  className="w-full border border-gray-300 rounded px-2 py-1.5 text-sm"
                                  value={(customerBudgetDraft as any)[key]}
                                  onChange={(e) =>
                                    setHealthCustomerBudgetDraft(customerRow.customer, {
                                      ...customerBudgetDraft,
                                      [key]: Math.max(0, Number(e.target.value || 0)),
                                    })
                                  }
                                />
                              </label>
                            ))}
                          </div>
                        </div>
                        {customerRow.rebalance_guidance_status === 'actionable' ? (
                          <div className="mt-3 border border-amber-200 rounded p-3 bg-amber-50">
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <div className="text-xs font-medium text-amber-900">Rebalance guidance</div>
                                <div className="text-[11px] text-amber-800 mt-1">
                                  {customerRow.rebalance_guidance_summary || 'Redistribute monitor-local caps to reduce portfolio pressure.'}
                                </div>
                                {(customerRow.rebalance_guidance_reasons || []).length > 0 ? (
                                  <div className="mt-2 flex flex-wrap gap-2">
                                    {customerRow.rebalance_guidance_reasons.map((reason) => (
                                      <span key={`${customerRow.customer}-rebalance-reason-${reason}`} className="text-[11px] bg-white border border-amber-200 text-amber-900 px-2 py-1 rounded">
                                        {reason}
                                      </span>
                                    ))}
                                  </div>
                                ) : null}
                              </div>
                              <div className="flex gap-2 shrink-0">
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  disabled={previewCustomerRebalanceMutation.isLoading || rebalanceUpdates.length === 0}
                                  onClick={() => {
                                    previewCustomerRebalanceMutation.mutate({
                                      customer: customerRow.customer,
                                      monitorBudgetUpdates: rebalanceUpdates,
                                    });
                                  }}
                                >
                                  Preview rebalance
                                </Button>
                                <Button
                                  size="sm"
                                  disabled={applyCustomerRebalanceMutation.isLoading || rebalanceUpdates.length === 0}
                                  onClick={() => {
                                    const ok = window.confirm(`Apply customer rebalance guidance for ${customerRow.customer}?`);
                                    if (!ok) return;
                                    applyCustomerRebalanceMutation.mutate({
                                      customer: customerRow.customer,
                                      monitorBudgetUpdates: rebalanceUpdates,
                                      changeReason: customerRow.rebalance_guidance_summary || `Customer rebalance guidance for ${customerRow.customer}`,
                                    });
                                  }}
                                >
                                  Apply rebalance
                                </Button>
                              </div>
                            </div>
                            <div className="mt-3 space-y-2">
                              {(customerRow.rebalance_guidance_changes || []).map((change) => (
                                <div key={`${customerRow.customer}-rebalance-${change.monitor_job_id}`} className="rounded border border-amber-200 bg-white px-3 py-2 text-xs text-amber-900">
                                  <div className="font-medium">{change.monitor_name}</div>
                                  <div className="mt-1">
                                    Auto {change.current_budget.auto_launch_limit_24h}→{change.proposed_budget.auto_launch_limit_24h}
                                    {' · '}
                                    Queue {change.current_budget.approval_queue_limit_24h}→{change.proposed_budget.approval_queue_limit_24h}
                                    {' · '}
                                    Alerts {change.current_budget.alert_limit_24h}→{change.proposed_budget.alert_limit_24h}
                                    {' · '}
                                    Backlog {change.current_budget.queue_backlog_cap}→{change.proposed_budget.queue_backlog_cap}
                                  </div>
                                  {(change.reasons || []).length > 0 ? (
                                    <div className="mt-1 text-[11px] text-amber-800">{change.reasons.join(' · ')}</div>
                                  ) : null}
                                </div>
                              ))}
                            </div>
                            {rebalancePreview?.customer === customerRow.customer ? (
                              <div className="mt-3 rounded border border-slate-200 bg-white p-3 text-xs">
                                <div className="font-medium text-slate-800">Rebalance preview</div>
                                <div className="mt-1 text-slate-600">
                                  Capacity before:
                                  {' '}
                                  Auto {rebalancePreview.before_capacity.auto_launch_limit_24h}
                                  {' · '}
                                  Queue {rebalancePreview.before_capacity.approval_queue_limit_24h}
                                  {' · '}
                                  Alerts {rebalancePreview.before_capacity.alert_limit_24h}
                                  {' · '}
                                  Backlog {rebalancePreview.before_capacity.queue_backlog_cap}
                                </div>
                                <div className="text-slate-600">
                                  Capacity after:
                                  {' '}
                                  Auto {rebalancePreview.after_capacity.auto_launch_limit_24h}
                                  {' · '}
                                  Queue {rebalancePreview.after_capacity.approval_queue_limit_24h}
                                  {' · '}
                                  Alerts {rebalancePreview.after_capacity.alert_limit_24h}
                                  {' · '}
                                  Backlog {rebalancePreview.after_capacity.queue_backlog_cap}
                                </div>
                                <div className="mt-2 space-y-2">
                                  {(rebalancePreview.changes || []).map((change) => (
                                    <div key={`${rebalancePreview.customer}-preview-${change.monitor_job_id}`} className="rounded bg-slate-50 px-2 py-2">
                                      <div className="font-medium text-slate-800">{change.monitor_name}</div>
                                      <div className="text-slate-600 mt-1">
                                        Auto {change.delta_budget.auto_launch_limit_24h >= 0 ? '+' : ''}{change.delta_budget.auto_launch_limit_24h}
                                        {' · '}
                                        Queue {change.delta_budget.approval_queue_limit_24h >= 0 ? '+' : ''}{change.delta_budget.approval_queue_limit_24h}
                                        {' · '}
                                        Alerts {change.delta_budget.alert_limit_24h >= 0 ? '+' : ''}{change.delta_budget.alert_limit_24h}
                                        {' · '}
                                        Backlog {change.delta_budget.queue_backlog_cap >= 0 ? '+' : ''}{change.delta_budget.queue_backlog_cap}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null}
                          </div>
                        ) : null}
                        <div className="mt-3 border border-gray-200 rounded p-3 bg-white">
                          <div className="flex items-center justify-between gap-2 mb-3">
                            <div>
                              <div className="text-xs font-medium text-gray-700">Rebalance history</div>
                              <div className="text-[11px] text-gray-500">
                                {(customerRow.recent_rebalance_history || []).length} recorded rebalance{(customerRow.recent_rebalance_history || []).length === 1 ? '' : 's'}
                              </div>
                            </div>
                          </div>
                          {(customerRow.recent_rebalance_history || []).length === 0 ? (
                            <p className="text-xs text-gray-500">No recorded rebalance actions yet.</p>
                          ) : (
                            <div className="space-y-2">
                              {(customerRow.recent_rebalance_history || []).map((entry) => {
                                const evaluationKey = `${customerRow.customer}:${entry.id}`;
                                const evaluationDetail = healthCustomerRebalanceEvaluations[evaluationKey];
                                return (
                                  <div key={entry.id} className="border border-gray-200 rounded p-3 bg-gray-50">
                                    <div className="flex items-start justify-between gap-3">
                                      <div>
                                        <div className="text-sm text-gray-900">
                                          {(entry.changes || []).map((change) => change.monitor_name).join(', ') || 'Customer rebalance'}
                                        </div>
                                        <div className="text-[11px] text-gray-500 mt-1">
                                          {formatPolicyHistoryTimestamp(entry.at)}
                                          {entry.change_source ? ` via ${formatPolicyChangeSource(entry.change_source)}` : ''}
                                          {entry.actor_user_id ? ` by ${entry.actor_user_id}` : ''}
                                        </div>
                                        {entry.change_reason ? (
                                          <div className="text-[11px] text-gray-600 mt-1">{entry.change_reason}</div>
                                        ) : null}
                                        {entry.evaluation_status ? (
                                          <div className="mt-2 flex flex-wrap gap-2">
                                            <span className={`text-[11px] px-2 py-1 rounded ${
                                              entry.evaluation_status === 'improving'
                                                ? 'bg-emerald-100 text-emerald-700'
                                                : entry.evaluation_status === 'degrading'
                                                  ? 'bg-rose-100 text-rose-700'
                                                  : entry.evaluation_status === 'mixed'
                                                    ? 'bg-amber-100 text-amber-800'
                                                    : 'bg-slate-100 text-slate-700'
                                            }`}>
                                              {formatPolicyEvaluationStatus(entry.evaluation_status)}
                                            </span>
                                            <span className="text-[11px] bg-white border border-gray-200 text-gray-700 px-2 py-1 rounded">
                                              Sample {entry.evaluation_sample_count}/{entry.evaluation_target_count || entry.evaluation_sample_count}
                                            </span>
                                            <span className="text-[11px] bg-white border border-gray-200 text-gray-700 px-2 py-1 rounded">
                                              Backlog {formatSimulationCountDelta(entry.delta_counts?.backlog_used || 0)} · Throttled {formatSimulationCountDelta(entry.delta_counts?.throttled_monitor_count || 0)} · Blocked {formatSimulationCountDelta(entry.delta_counts?.blocked_count || 0)}
                                            </span>
                                          </div>
                                        ) : null}
                                      </div>
                                      <div className="flex flex-col gap-2 shrink-0">
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          disabled={loadCustomerRebalanceEvaluationMutation.isLoading}
                                          onClick={() =>
                                            loadCustomerRebalanceEvaluationMutation.mutate({
                                              customer: customerRow.customer,
                                              historyEntryId: entry.id,
                                            })
                                          }
                                        >
                                          Compare outcome
                                        </Button>
                                      </div>
                                    </div>
                                    {evaluationDetail ? (
                                      <div className="mt-3 border border-slate-200 rounded bg-white p-3">
                                        <div className="flex items-center gap-2 flex-wrap">
                                          <span className={`text-[11px] px-2 py-1 rounded ${
                                            evaluationDetail.evaluation_status === 'improving'
                                              ? 'bg-emerald-100 text-emerald-700'
                                              : evaluationDetail.evaluation_status === 'degrading'
                                                ? 'bg-rose-100 text-rose-700'
                                                : evaluationDetail.evaluation_status === 'mixed'
                                                  ? 'bg-amber-100 text-amber-800'
                                                  : 'bg-slate-100 text-slate-700'
                                          }`}>
                                            {formatPolicyEvaluationStatus(evaluationDetail.evaluation_status)}
                                          </span>
                                          <span className="text-[11px] text-slate-600">
                                            {evaluationDetail.evaluation_sample_count}/{evaluationDetail.evaluation_target_count} accepted signals after rebalance
                                          </span>
                                        </div>
                                        <div className="grid grid-cols-3 gap-3 mt-3 text-[11px]">
                                          <div className="border border-slate-200 rounded p-2">
                                            <div className="font-medium text-slate-700">Before</div>
                                            <div className="mt-1 text-slate-600">
                                              Backlog {evaluationDetail.before_counts.backlog_used} · Throttled {evaluationDetail.before_counts.throttled_monitor_count} · Blocked {evaluationDetail.before_counts.blocked_count}
                                            </div>
                                          </div>
                                          <div className="border border-slate-200 rounded p-2">
                                            <div className="font-medium text-slate-700">After</div>
                                            <div className="mt-1 text-slate-600">
                                              Backlog {evaluationDetail.after_counts.backlog_used} · Throttled {evaluationDetail.after_counts.throttled_monitor_count} · Blocked {evaluationDetail.after_counts.blocked_count}
                                            </div>
                                          </div>
                                          <div className="border border-slate-200 rounded p-2">
                                            <div className="font-medium text-slate-700">Delta</div>
                                            <div className="mt-1 text-slate-600">
                                              Backlog {formatSimulationCountDelta(evaluationDetail.delta_counts.backlog_used)} · Throttled {formatSimulationCountDelta(evaluationDetail.delta_counts.throttled_monitor_count)} · Blocked {formatSimulationCountDelta(evaluationDetail.delta_counts.blocked_count)}
                                            </div>
                                          </div>
                                        </div>
                                        {(evaluationDetail.evaluation_reasons || []).length > 0 ? (
                                          <div className="mt-3 flex flex-wrap gap-2">
                                            {evaluationDetail.evaluation_reasons.map((reason) => (
                                              <span key={reason} className="text-[11px] bg-slate-50 text-slate-700 border border-slate-200 px-2 py-1 rounded">
                                                {reason}
                                              </span>
                                            ))}
                                          </div>
                                        ) : null}
                                        {(evaluationDetail.sample_items || []).length > 0 ? (
                                          <div className="mt-3 space-y-2">
                                            <div className="text-[11px] font-medium text-slate-700">Sample signals</div>
                                            {evaluationDetail.sample_items.map((sample) => (
                                              <div key={`${sample.period}-${sample.item_id}`} className="border border-slate-200 rounded p-2">
                                                <div className="flex items-start justify-between gap-3">
                                                  <div className="min-w-0">
                                                    <div className="text-xs font-medium text-slate-900">{sample.title}</div>
                                                    <div className="text-[11px] text-slate-600 mt-1">
                                                      {sample.period} · {sample.monitor_name || 'Unknown monitor'}
                                                      {sample.launch_status ? ` · ${sample.launch_status.replace(/_/g, ' ')}` : ''}
                                                      {sample.outcome_status ? ` · ${sample.outcome_status.replace(/_/g, ' ')}` : ''}
                                                    </div>
                                                    {sample.summary ? (
                                                      <div className="text-[11px] text-slate-600 mt-1">{sample.summary}</div>
                                                    ) : null}
                                                  </div>
                                                  <Button
                                                    size="sm"
                                                    variant="ghost"
                                                    onClick={() => {
                                                      setActiveTab('inbox');
                                                      setInboxCustomerFilter(customerRow.customer);
                                                      setInboxStatusFilter('accepted');
                                                      setInboxTypeFilter('');
                                                      setInboxSearch('');
                                                      setInboxJobFilter(sample.monitor_job_id ? String(sample.monitor_job_id) : '');
                                                      const params = new URLSearchParams(location.search);
                                                      params.set('tab', 'inbox');
                                                      params.set('customer', customerRow.customer);
                                                      if (sample.monitor_job_id) params.set('inbox_job', String(sample.monitor_job_id));
                                                      params.set('inbox', sample.item_id);
                                                      navigate(`${location.pathname}?${params.toString()}`, { replace: true });
                                                    }}
                                                  >
                                                    Open in Inbox
                                                  </Button>
                                                </div>
                                              </div>
                                            ))}
                                          </div>
                                        ) : null}
                                      </div>
                                    ) : null}
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3 text-xs">
                          <div className="bg-slate-50 rounded p-3">
                            <div className="font-medium text-slate-700 mb-1">Top launch monitors</div>
                            {(customerRow.top_launch_monitors || []).length > 0 ? (
                              <div className="space-y-1">
                                {customerRow.top_launch_monitors.map((row) => (
                                  <div key={`${customerRow.customer}-launch-${row.monitor_name}`} className="text-slate-600">
                                    {row.monitor_name} · {row.value}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-slate-500">No recent launch pressure.</div>
                            )}
                          </div>
                          <div className="bg-slate-50 rounded p-3">
                            <div className="font-medium text-slate-700 mb-1">Top backlog / alerts</div>
                            <div className="space-y-1">
                              {(customerRow.top_backlog_monitors || []).slice(0, 2).map((row) => (
                                <div key={`${customerRow.customer}-backlog-${row.monitor_name}`} className="text-slate-600">
                                  Backlog: {row.monitor_name} · {row.value}
                                </div>
                              ))}
                              {(customerRow.top_alert_monitors || []).slice(0, 1).map((row) => (
                                <div key={`${customerRow.customer}-alert-${row.monitor_name}`} className="text-slate-600">
                                  Alerts: {row.monitor_name} · {row.value}
                                </div>
                              ))}
                              {(customerRow.throttled_monitors || []).slice(0, 1).map((row) => (
                                <div key={`${customerRow.customer}-throttle-${row.monitor_name}`} className="text-amber-700">
                                  Throttled: {row.monitor_name} · {String(row.throttle_state || '').replace(/_/g, ' ')}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-2 mb-4">
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={healthCustomerFilter}
                onChange={(e) => setHealthCustomerFilter(e.target.value)}
              >
                <option value="">All customers</option>
                {healthCustomers.map((customer) => (
                  <option key={customer} value={customer}>
                    {customer}
                  </option>
                ))}
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={healthMonitorTypeFilter}
                onChange={(e) => setHealthMonitorTypeFilter(e.target.value)}
              >
                <option value="">All monitor types</option>
                <option value="monitor">Monitor</option>
                <option value="research">Research</option>
                <option value="analysis">Analysis</option>
                <option value="custom">Custom</option>
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={healthBucketFilter}
                onChange={(e) => setHealthBucketFilter(e.target.value)}
              >
                <option value="">All health buckets</option>
                <option value="strong">Strong</option>
                <option value="mixed">Mixed</option>
                <option value="weak">Weak</option>
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={healthAutonomyFilter}
                onChange={(e) => setHealthAutonomyFilter(e.target.value)}
              >
                <option value="">All autonomy modes</option>
                <option value="auto_launch_safe">Auto launch safe</option>
                <option value="queue_for_approval">Queue for approval</option>
                <option value="manual_only">Manual only</option>
              </select>
              {(healthCustomerFilter || healthMonitorTypeFilter || healthBucketFilter || healthAutonomyFilter) && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setHealthCustomerFilter('');
                    setHealthMonitorTypeFilter('');
                    setHealthBucketFilter('');
                    setHealthAutonomyFilter('');
                  }}
                >
                  <XCircle className="w-4 h-4 mr-1" />
                  Clear Filters
                </Button>
              )}
            </div>

            {deepLinkedHealthMonitor ? (
              <div className="flex items-center gap-2 mb-4 text-xs">
                <span className="bg-cyan-50 text-cyan-800 border border-cyan-200 px-2 py-1 rounded">
                  Showing {healthCustomerFilter || 'all'} monitors · focused on {(filteredMonitorAnalytics.monitors.find((monitor) => String(monitor.monitor_job_id || '').trim() === String(deepLinkedHealthMonitor || '').trim())?.monitor_name || deepLinkedHealthMonitor)}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    navigate(buildAutonomousAgentsUrl(undefined, {
                      health_monitor: null,
                    }), { replace: true });
                  }}
                >
                  Clear focus
                </Button>
              </div>
            ) : null}

            {monitorAnalyticsLoading ? (
              <div className="flex justify-center items-center flex-1">
                <LoadingSpinner />
              </div>
            ) : filteredMonitorAnalytics.monitors.length === 0 ? (
              <div className="flex flex-col items-center justify-center flex-1 text-gray-500">
                <Activity className="w-12 h-12 mb-3 text-gray-400" />
                <p className="text-lg font-medium">No monitor analytics yet</p>
                <p className="text-sm">Accept or reject inbox items to start building monitor health data.</p>
              </div>
            ) : (
              <div className="grid grid-cols-[minmax(0,2fr)_minmax(320px,1fr)] gap-4 min-h-0 flex-1">
                <div className="space-y-3 overflow-y-auto pr-1">
                  {filteredMonitorAnalytics.monitors.map((monitor: ResearchMonitorHealthSummary) => {
                    const monitorJobId = String(monitor.monitor_job_id || '').trim();
                    const policyDraft = getHealthPolicyDraft(monitor);
                    const budgetDraft = getHealthBudgetDraft(monitor);
                    const recommendationChoices = Array.from(
                      new Set([
                        'deep_dive_chain',
                        'single_research_job',
                        ...canonicalAllowedRecommendationsFromMonitor(monitor),
                        ...(monitor.recommended_allowed_recommendations || []),
                        ...(monitor.top_recommendations || []).map((recommendation) => recommendation.recommendation_key),
                      ].filter(Boolean))
                    );
                    const currentReviewMode = canonicalReviewModeFromMonitor(monitor);
                    const currentAllowedRecommendations = canonicalAllowedRecommendationsFromMonitor(monitor);
                    const policyChanged =
                      policyDraft.automation_profile !== String(monitor.automation_profile || monitor.autonomy_mode || 'balanced') ||
                      policyDraft.mode !== currentReviewMode ||
                      JSON.stringify([...policyDraft.allowed].sort()) !==
                        JSON.stringify([...currentAllowedRecommendations].sort());
                    const matchesRecommendedPolicy =
                      policyDraft.automation_profile === String(monitor.automation_profile || monitor.autonomy_mode || 'balanced') &&
                      policyDraft.mode === String(monitor.recommended_policy_mode || 'manual_only') &&
                      JSON.stringify([...policyDraft.allowed].sort()) ===
                        JSON.stringify([...(monitor.recommended_allowed_recommendations || [])].sort());
                    const budgetChanged =
                      budgetDraft.auto_launch_limit_24h !== Number(monitor.autonomy_budget?.auto_launch_limit_24h || 0)
                      || budgetDraft.approval_queue_limit_24h !== Number(monitor.autonomy_budget?.approval_queue_limit_24h || 0)
                      || budgetDraft.alert_limit_24h !== Number(monitor.autonomy_budget?.alert_limit_24h || 0)
                      || budgetDraft.queue_backlog_cap !== Number(monitor.autonomy_budget?.queue_backlog_cap || 0);
                    const policySimulation = monitorJobId ? healthPolicySimulations[monitorJobId] : undefined;
                    const latestEvaluationStatus = String(monitor.latest_policy_evaluation_status || '').trim();

                    return (
                    <div
                      ref={registerHealthMonitorCardRef(monitorJobId)}
                      key={`${monitor.monitor_job_id || 'unattributed'}-${monitor.customer || 'global'}`}
                      className={`bg-white border rounded-lg p-4 ${deepLinkedHealthMonitor && deepLinkedHealthMonitor === monitorJobId ? HEALTH_FOCUS_CARD_CLASS : 'border-gray-200'}`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <div className="flex items-center gap-2 flex-wrap">
                            <h3 className="section-heading">{monitor.monitor_name}</h3>
                            <span
                              className={`text-xs px-2 py-1 rounded ${
                                monitor.health_bucket === 'strong'
                                  ? 'bg-emerald-100 text-emerald-700'
                                  : monitor.health_bucket === 'mixed'
                                    ? 'bg-amber-100 text-amber-800'
                                    : 'bg-rose-100 text-rose-700'
                              }`}
                            >
                              {monitor.health_bucket}
                            </span>
                            {monitor.customer ? (
                              <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">{monitor.customer}</span>
                            ) : null}
                            {monitor.monitor_job_type ? (
                              <span className="text-xs bg-slate-100 text-slate-700 px-2 py-1 rounded">{monitor.monitor_job_type}</span>
                            ) : null}
                            {latestEvaluationStatus ? (
                              <span
                                className={`text-xs px-2 py-1 rounded ${
                                  latestEvaluationStatus === 'improving'
                                    ? 'bg-emerald-100 text-emerald-700'
                                    : latestEvaluationStatus === 'degrading'
                                      ? 'bg-rose-100 text-rose-700'
                                      : latestEvaluationStatus === 'mixed'
                                        ? 'bg-amber-100 text-amber-800'
                                        : 'bg-slate-100 text-slate-700'
                                }`}
                              >
                                Policy {formatPolicyEvaluationStatus(latestEvaluationStatus)}
                              </span>
                            ) : null}
                          </div>
                          <p className="text-sm text-gray-500 mt-1">
                            Health score {monitor.health_score.toFixed(1)} · Acceptance {monitor.acceptance_rate.toFixed(1)}%
                          </p>
                          <div className="mt-2 flex flex-wrap gap-2">
                            <span className={`text-[11px] px-2 py-1 rounded ${
                              monitor.budget_throttle_state === 'normal'
                                ? 'bg-emerald-100 text-emerald-700'
                                : monitor.budget_throttle_state === 'auto_launch_throttled'
                                  ? 'bg-amber-100 text-amber-800'
                                  : 'bg-rose-100 text-rose-700'
                            }`}>
                              Budget {String(monitor.budget_throttle_state || 'normal').replace(/_/g, ' ')}
                            </span>
                            <span className="text-[11px] bg-slate-100 text-slate-700 px-2 py-1 rounded">
                              Auto {monitor.budget_usage?.auto_launch_count_24h || 0}/{monitor.autonomy_budget?.auto_launch_limit_24h || 0}
                            </span>
                            <span className="text-[11px] bg-slate-100 text-slate-700 px-2 py-1 rounded">
                              Queue {monitor.budget_usage?.approval_queue_count_24h || 0}/{monitor.autonomy_budget?.approval_queue_limit_24h || 0}
                            </span>
                            <span className="text-[11px] bg-slate-100 text-slate-700 px-2 py-1 rounded">
                              Backlog {monitor.budget_usage?.queue_backlog_count || 0}/{monitor.autonomy_budget?.queue_backlog_cap || 0}
                            </span>
                          </div>
                          {(monitor.budget_throttle_reasons || []).length > 0 ? (
                            <div className="mt-2 flex flex-wrap gap-2">
                              {(monitor.budget_throttle_reasons || []).map((reason) => (
                                <span key={reason} className="text-[11px] bg-amber-50 text-amber-800 px-2 py-1 rounded border border-amber-200">
                                  {reason}
                                </span>
                              ))}
                            </div>
                          ) : null}
                          {latestEvaluationStatus ? (
                            <div className="mt-2 flex flex-wrap gap-2">
                              <span className="text-[11px] bg-slate-100 text-slate-700 px-2 py-1 rounded">
                                Post-change sample {monitor.latest_policy_evaluation_sample_count}/{monitor.latest_policy_evaluation_target_count || monitor.latest_policy_evaluation_sample_count}
                              </span>
                              {(monitor.latest_policy_evaluation_reasons || []).map((reason) => (
                                <span key={reason} className="text-[11px] bg-slate-50 text-slate-700 px-2 py-1 rounded border border-slate-200">
                                  {reason}
                                </span>
                              ))}
                            </div>
                          ) : null}
                        </div>
                        <div className="flex gap-2 shrink-0">
                          <Button
                            size="sm"
                            variant="ghost"
                              onClick={() => {
                                setActiveTab('inbox');
                                setInboxStatusFilter('');
                                setInboxTypeFilter('');
                                setInboxSearch('');
                                setInboxHealthDrilldown('');
                                setInboxPolicyDrilldown('');
                              }}
                            >
                              View Inbox
                          </Button>
                          {monitor.customer ? (
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => {
                                setActiveTab('queue');
                                setQueueCustomerFilter(monitor.customer || '');
                                setQueueJobFilter('');
                                setQueueHealthDrilldown('');
                              }}
                            >
                              View Queue
                            </Button>
                          ) : null}
                        </div>
                      </div>

                      <div className="grid grid-cols-4 gap-3 mt-4 text-sm">
                        <div className="bg-gray-50 rounded p-3">
                          <div className="text-xs uppercase tracking-wide text-gray-500">Discovery</div>
                          <div className="font-semibold text-gray-900 mt-1">{monitor.discovered_count}</div>
                          <div className="text-xs text-gray-500 mt-1">Accepted {monitor.accepted_count} · Rejected {monitor.rejected_count}</div>
                        </div>
                        <div className="bg-gray-50 rounded p-3">
                          <div className="text-xs uppercase tracking-wide text-gray-500">Launches</div>
                          <div className="font-semibold text-gray-900 mt-1">{monitor.auto_launched_count + monitor.approval_launched_count}</div>
                          <div className="text-xs text-gray-500 mt-1">Auto {monitor.auto_launched_count} · Approved {monitor.approval_launched_count}</div>
                        </div>
                        <div className="bg-gray-50 rounded p-3">
                          <div className="text-xs uppercase tracking-wide text-gray-500">Outcomes</div>
                          <div className="mt-1 flex flex-wrap gap-2">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => openInboxHealthDrilldown('completed_follow_up', { customer: monitor.customer, monitorJobId })}
                            >
                              Completed {monitor.follow_up_completed_count}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => openInboxHealthDrilldown('failed_follow_up', { customer: monitor.customer, monitorJobId })}
                            >
                              Failed {monitor.follow_up_failed_count}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => openInboxHealthDrilldown('cancelled_follow_up', { customer: monitor.customer, monitorJobId })}
                            >
                              Cancelled {monitor.follow_up_cancelled_count}
                            </Button>
                          </div>
                        </div>
                        <div className="bg-gray-50 rounded p-3">
                          <div className="text-xs uppercase tracking-wide text-gray-500">Policy drag</div>
                          <div className="font-semibold text-gray-900 mt-1">{monitor.blocked_count}</div>
                          <div className="text-xs text-gray-500 mt-1">
                            Manual {monitor.manual_only_count} · Pending {monitor.queued_for_approval_count} · Relaunch {monitor.relaunch_count}
                          </div>
                          <div className="mt-1 flex flex-wrap gap-2">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => openQueueHealthDrilldown('blocked_follow_up', { customer: monitor.customer, monitorJobId })}
                            >
                              Blocked {monitor.blocked_count}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => openQueueHealthDrilldown('manual_follow_up_recommendations', { customer: monitor.customer, monitorJobId })}
                            >
                              Manual {monitor.manual_only_count}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => openQueueHealthDrilldown('pending_follow_up_approvals', { customer: monitor.customer, monitorJobId })}
                            >
                              Queue {monitor.queued_for_approval_count}
                            </Button>
                          </div>
                        </div>
                      </div>

                      <div className="mt-3 text-xs text-gray-600 flex flex-wrap gap-2">
                        {(monitor.health_reasons || []).map((reason) => (
                          <span key={reason} className="bg-primary-50 text-primary-700 px-2 py-1 rounded">
                            {reason}
                          </span>
                        ))}
                      </div>

                      <div className="mt-4 grid grid-cols-2 gap-4">
                        <div className="border border-gray-200 rounded p-3 bg-slate-50">
                          <div className="text-xs font-medium text-gray-700 mb-2">Effective autonomy</div>
                          <div className="text-sm text-gray-900">
                            Review mode: <span className="font-medium">{formatReviewModeLabel(currentReviewMode)}</span>
                          </div>
                          <div className="text-xs text-gray-500 mt-2">
                            Autonomy: {formatAutonomyLabel(monitor.autonomy_mode || monitor.automation_profile || 'balanced')}
                            {' · '}
                            Effective review {formatReviewModeLabel(currentReviewMode)}
                          </div>
                          <div className="text-xs text-gray-500 mt-2">
                            Allowlist: {currentAllowedRecommendations.join(', ') || 'None'}
                          </div>
                          {monitor.scheduler_summary ? (
                            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-gray-500">
                              <span>
                                Queue {Number(monitor.scheduler_summary.queued_approvals_count || 0)}
                              </span>
                              <span>
                                Manual {Number(monitor.scheduler_summary.manual_recommendations_count || 0)}
                              </span>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => openInboxHealthDrilldown('suppressed_relaunch', { customer: monitor.customer, monitorJobId })}
                              >
                                Suppressed {Number(monitor.scheduler_summary.suppressed_relaunches_count || monitor.suppressed_relaunches_count || 0)}
                              </Button>
                            </div>
                          ) : null}
                          {monitor.budget_clamp_state ? (
                            <div className="text-[11px] text-amber-700 mt-2">
                              Budget clamp {String(monitor.budget_clamp_state).replace(/_/g, ' ')}
                            </div>
                          ) : null}
                          {monitor.latest_policy_changed_at ? (
                            <div className="text-[11px] text-gray-500 mt-2">
                              Last changed {formatPolicyHistoryTimestamp(monitor.latest_policy_changed_at)}
                              {monitor.latest_policy_change_source ? ` via ${formatPolicyChangeSource(monitor.latest_policy_change_source)}` : ''}
                            </div>
                          ) : null}
                        </div>
                        <div className="border border-gray-200 rounded p-3 bg-emerald-50">
                          <div className="flex items-center justify-between gap-2">
                            <div className="text-xs font-medium text-emerald-800">Recommended policy</div>
                            <span className="text-[10px] uppercase tracking-wide text-emerald-700">
                              {monitor.policy_confidence} confidence
                            </span>
                          </div>
                          <div className="text-sm text-emerald-900 mt-2">
                            Mode: <span className="font-medium">{monitor.recommended_policy_mode.replace(/_/g, ' ')}</span>
                          </div>
                          <div className="text-xs text-emerald-800 mt-2">
                            Allowlist: {(monitor.recommended_allowed_recommendations || []).join(', ') || 'None'}
                          </div>
                          {(monitor.policy_reasons || []).length > 0 ? (
                            <div className="mt-2 flex flex-wrap gap-2">
                              {monitor.policy_reasons.map((reason) => (
                                <span key={reason} className="text-[11px] bg-white/70 text-emerald-900 px-2 py-1 rounded">
                                  {reason}
                                </span>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      </div>

                      {monitor.policy_guardrail_status === 'active' && monitorJobId ? (
                        <div className="mt-4 border border-rose-200 rounded p-3 bg-rose-50">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <div className="text-xs font-medium text-rose-800">Policy safeguard recommended</div>
                              <div className="text-sm text-rose-900 mt-1">
                                Suggested action: {String(monitor.policy_guardrail_action || 'review').replace(/_/g, ' ')}
                                {monitor.policy_guardrail_target_policy?.follow_up_review_mode || monitor.policy_guardrail_follow_up_autonomy?.mode ? (
                                  <span> to <span className="font-medium">{String(monitor.policy_guardrail_target_policy?.follow_up_review_mode || monitor.policy_guardrail_follow_up_autonomy?.mode).replace(/_/g, ' ')}</span></span>
                                ) : null}
                              </div>
                              {(monitor.policy_guardrail_reasons || []).length > 0 ? (
                                <div className="mt-2 flex flex-wrap gap-2">
                                  {monitor.policy_guardrail_reasons.map((reason) => (
                                    <span key={reason} className="text-[11px] bg-white/70 text-rose-900 px-2 py-1 rounded">
                                      {reason}
                                    </span>
                                  ))}
                                </div>
                              ) : null}
                            </div>
                            <div className="flex flex-col gap-2 shrink-0">
                              <Button
                                size="sm"
                                disabled={updateMonitorPolicyMutation.isLoading || rollbackMonitorPolicyMutation.isLoading}
                                onClick={() => {
                                  if (monitor.policy_guardrail_action === 'rollback' && monitor.policy_guardrail_target_history_entry_id) {
                                    rollbackMonitorPolicyMutation.mutate({
                                      monitorJobId,
                                      historyEntryId: monitor.policy_guardrail_target_history_entry_id,
                                    });
                                    return;
                                  }
                                  updateMonitorPolicyMutation.mutate({
                                    monitorJobId,
                                    data: {
                                      automation_profile: String(monitor.automation_profile || monitor.autonomy_mode || 'balanced'),
                                      automation_policy: {
                                        follow_up_review_mode: String(monitor.policy_guardrail_target_policy?.follow_up_review_mode || monitor.policy_guardrail_follow_up_autonomy?.mode || 'manual_only'),
                                        allowed_recommendations: (monitor.policy_guardrail_target_policy?.allowed_recommendations || monitor.policy_guardrail_follow_up_autonomy?.allowed_recommendations || []),
                                      },
                                      change_source: 'policy_guardrail',
                                      change_reason: 'Applied degrading-policy safeguard from autonomy health',
                                      analytics_context: getHealthPolicyAnalyticsContext(monitor),
                                    },
                                  });
                                }}
                              >
                                Apply safeguard
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => openHealthPolicyComparison(monitorJobId, monitor.policy_guardrail_target_history_entry_id || monitor.recent_policy_history?.[0]?.id)}
                              >
                                Compare before/after
                              </Button>
                            </div>
                          </div>
                        </div>
                      ) : null}

                      {monitorJobId ? (
                        <div className="mt-4 border border-gray-200 rounded p-3">
                          <div className="text-xs font-medium text-gray-700 mb-3">Autonomy controls</div>
                          <div className="grid grid-cols-[200px_minmax(0,1fr)] gap-4 items-start">
                            <div>
                              <label className="block text-xs font-medium text-gray-600 mb-1">Automation profile</label>
                              <select
                                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                                value={policyDraft.automation_profile}
                                onChange={(e) =>
                                  setHealthPolicyDraft(monitorJobId, {
                                    ...policyDraft,
                                    automation_profile: e.target.value,
                                  })
                                }
                              >
                                <option value="balanced">Balanced</option>
                                <option value="max_autonomy">Max autonomy</option>
                              </select>
                            </div>
                            <div>
                              <label className="block text-xs font-medium text-gray-600 mb-1">Review mode</label>
                              <select
                                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                                value={policyDraft.mode}
                                onChange={(e) =>
                                  setHealthPolicyDraft(monitorJobId, {
                                    ...policyDraft,
                                    mode: e.target.value,
                                  })
                                }
                              >
                                <option value="manual_only">Manual only</option>
                                <option value="queue_for_approval">Queue for approval</option>
                                <option value="auto_launch_safe">Auto launch safe</option>
                              </select>
                            </div>
                            <div>
                              <div className="text-xs font-medium text-gray-600 mb-1">Allowed recommendations</div>
                              <div className="flex flex-wrap gap-2">
                                {recommendationChoices.map((key) => {
                                  const checked = policyDraft.allowed.includes(key);
                                  return (
                                    <label key={key} className="inline-flex items-center gap-2 text-xs text-gray-700 bg-gray-50 border border-gray-200 rounded px-2 py-1">
                                      <input
                                        type="checkbox"
                                        checked={checked}
                                        onChange={(e) => {
                                          const nextAllowed = e.target.checked
                                            ? Array.from(new Set([...policyDraft.allowed, key]))
                                            : policyDraft.allowed.filter((value) => value !== key);
                                          setHealthPolicyDraft(monitorJobId, {
                                            ...policyDraft,
                                            allowed: nextAllowed,
                                          });
                                        }}
                                      />
                                      <span>{key}</span>
                                    </label>
                                  );
                                })}
                              </div>
                            </div>
                          </div>
                          <div className="mt-3 flex flex-wrap items-center gap-2">
                            <Button
                              size="sm"
                              variant="secondary"
                              disabled={updateMonitorPolicyMutation.isLoading}
                              onClick={() =>
                                setHealthPolicyDraft(monitorJobId, {
                                  automation_profile: String(monitor.automation_profile || monitor.autonomy_mode || 'balanced'),
                                  mode: monitor.recommended_policy_mode,
                                  allowed: monitor.recommended_allowed_recommendations || [],
                                })
                              }
                            >
                              Use recommendation
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={simulateMonitorPolicyMutation.isLoading}
                              onClick={() =>
                                simulateMonitorPolicyMutation.mutate({
                                  monitorJobId,
                                  data: {
                                    automation_profile: policyDraft.automation_profile,
                                    automation_policy: {
                                      follow_up_review_mode: policyDraft.mode,
                                      allowed_recommendations: policyDraft.allowed,
                                    },
                                    mode: policyDraft.mode,
                                    allowed_recommendations: policyDraft.allowed,
                                    history_limit: 25,
                                  },
                                })
                              }
                            >
                              Preview impact
                            </Button>
                            <Button
                              size="sm"
                              disabled={!policyChanged || updateMonitorPolicyMutation.isLoading}
                              onClick={() =>
                                updateMonitorPolicyMutation.mutate({
                                  monitorJobId,
                                  data: {
                                    automation_profile: policyDraft.automation_profile,
                                    automation_policy: {
                                      follow_up_review_mode: policyDraft.mode,
                                      allowed_recommendations: policyDraft.allowed,
                                    },
                                    mode: policyDraft.mode,
                                    allowed_recommendations: policyDraft.allowed,
                                    change_source: matchesRecommendedPolicy ? 'guided_recommendation' : 'manual_override',
                                    analytics_context: getHealthPolicyAnalyticsContext(monitor),
                                  },
                                })
                              }
                            >
                              Apply policy
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={updateMonitorPolicyMutation.isLoading}
                              onClick={() =>
                                updateMonitorPolicyMutation.mutate({
                                  monitorJobId,
                                  data: {
                                    reset_to_default: true,
                                    change_source: 'reset_to_default',
                                    analytics_context: getHealthPolicyAnalyticsContext(monitor),
                                  },
                                })
                              }
                            >
                              Reset to default
                            </Button>
                          </div>
                        </div>
                      ) : null}

                      {monitorJobId ? (
                        <div className="mt-4 border border-amber-200 rounded p-3 bg-amber-50">
                          <div className="flex items-center justify-between gap-2 mb-3">
                            <div>
                              <div className="text-xs font-medium text-amber-900">Autonomy budgets</div>
                              <div className="text-[11px] text-amber-800">
                                Rolling 24h launch, approval, and alert caps plus active backlog control.
                              </div>
                            </div>
                            <span className="text-[11px] bg-white/70 text-amber-900 px-2 py-1 rounded">
                              State {String(monitor.budget_throttle_state || 'normal').replace(/_/g, ' ')}
                            </span>
                          </div>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {[
                              ['Auto launches', 'auto_launch_limit_24h', monitor.budget_usage?.auto_launch_count_24h || 0],
                              ['Approval queue', 'approval_queue_limit_24h', monitor.budget_usage?.approval_queue_count_24h || 0],
                              ['Alerts', 'alert_limit_24h', monitor.budget_usage?.alert_count_24h || 0],
                              ['Backlog cap', 'queue_backlog_cap', monitor.budget_usage?.queue_backlog_count || 0],
                            ].map(([label, key, used]) => (
                              <label key={String(key)} className="block">
                                <div className="text-xs font-medium text-amber-900">{label}</div>
                                <div className="text-[11px] text-amber-800 mb-1">Used {used}</div>
                                <input
                                  type="number"
                                  min={0}
                                  className="w-full border border-amber-300 rounded-lg px-3 py-2 text-sm bg-white"
                                  value={(budgetDraft as any)[key]}
                                  onChange={(e) =>
                                    setHealthBudgetDraft(monitorJobId, {
                                      ...budgetDraft,
                                      [key]: Math.max(0, Number(e.target.value || 0)),
                                    } as any)
                                  }
                                />
                              </label>
                            ))}
                          </div>
                          <div className="mt-3 flex flex-wrap items-center gap-2">
                            <Button
                              size="sm"
                              disabled={!budgetChanged || updateMonitorBudgetMutation.isLoading}
                              onClick={() =>
                                updateMonitorBudgetMutation.mutate({
                                  monitorJobId,
                                  data: { ...budgetDraft },
                                })
                              }
                            >
                              Apply budgets
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={updateMonitorBudgetMutation.isLoading}
                              onClick={() =>
                                updateMonitorBudgetMutation.mutate({
                                  monitorJobId,
                                  data: { reset_to_default: true },
                                })
                              }
                            >
                              Reset budgets
                            </Button>
                            <span className="text-[11px] text-amber-800">
                              Remaining auto {monitor.budget_remaining?.auto_launch_count_24h || 0} · queue {monitor.budget_remaining?.approval_queue_count_24h || 0} · alerts {monitor.budget_remaining?.alert_count_24h || 0}
                            </span>
                          </div>
                          {monitor.latest_budget_change_source ? (
                            <div className="mt-2 text-[11px] text-amber-800">
                              Latest budget change: {formatPolicyChangeSource(monitor.latest_budget_change_source)}
                              {monitor.latest_budget_change_reason ? ` · ${monitor.latest_budget_change_reason}` : ''}
                            </div>
                          ) : null}
                        </div>
                      ) : null}

                      {monitorJobId && policySimulation ? (
                        <div className="mt-4 border border-sky-200 rounded p-3 bg-sky-50">
                          <div className="flex items-center justify-between gap-2 mb-3">
                            <div>
                              <div className="text-xs font-medium text-sky-900">Policy impact preview</div>
                              <div className="text-[11px] text-sky-800">
                                Simulated from the last {policySimulation.history_limit} accepted inbox items.
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => openInboxForMonitorSignal(monitorJobId, undefined, 'simulated_policy_impact')}
                              >
                                View affected signals
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() =>
                                  setHealthPolicySimulations((prev) => {
                                    const next = { ...prev };
                                    delete next[monitorJobId];
                                    return next;
                                  })
                                }
                              >
                                Dismiss
                              </Button>
                            </div>
                          </div>
                          <div className="grid grid-cols-3 gap-3 text-xs">
                            {[
                              ['Auto launch', 'auto_launch_safe_count'],
                              ['Queue', 'queue_for_approval_count'],
                              ['Manual', 'manual_only_count'],
                              ['Blocked', 'blocked_count'],
                              ['Insufficient context', 'insufficient_context_count'],
                            ].map(([label, key]) => (
                              <div key={key} className="bg-white border border-sky-100 rounded p-3">
                                <div className="uppercase tracking-wide text-sky-700">{label}</div>
                                <div className="mt-2 text-slate-900">
                                  Current {(policySimulation.baseline_counts as any)[key]} {'->'} Proposed {(policySimulation.simulated_counts as any)[key]}
                                </div>
                                <div className="mt-1 text-sky-800">
                                  Delta {formatSimulationCountDelta(Number((policySimulation.delta_counts as any)[key] || 0))}
                                </div>
                              </div>
                            ))}
                          </div>
                          {(policySimulation.top_recommendation_deltas || []).length > 0 ? (
                            <div className="mt-3">
                              <div className="text-xs font-medium text-sky-900 mb-2">Top recommendation deltas</div>
                              <div className="flex flex-wrap gap-2">
                                {policySimulation.top_recommendation_deltas.map((row) => (
                                  <span key={row.recommendation_key} className="text-[11px] bg-white border border-sky-100 text-sky-900 px-2 py-1 rounded">
                                    {row.recommendation_key}: {row.baseline_count} {'->'} {row.simulated_count} ({formatSimulationCountDelta(row.delta_count)})
                                  </span>
                                ))}
                              </div>
                            </div>
                          ) : null}
                          {(policySimulation.sample_items || []).length > 0 ? (
                            <div className="mt-3 space-y-2">
                              <div className="text-xs font-medium text-sky-900">Sample item changes</div>
                              {policySimulation.sample_items.map((sample) => (
                                <div key={sample.item_id} className="bg-white border border-sky-100 rounded p-3">
                                  <div className="flex items-start justify-between gap-3">
                                    <div className="min-w-0">
                                      <div className="text-sm font-medium text-slate-900">{sample.title}</div>
                                      <div className="text-[11px] text-slate-600 mt-1">
                                        {sample.current_outcome.replace(/_/g, ' ')} {'->'} {sample.simulated_outcome.replace(/_/g, ' ')}
                                        {sample.recommendation_key ? ` via ${sample.recommendation_key}` : ''}
                                      </div>
                                      <div className="text-[11px] text-sky-900 mt-1">{sample.reason}</div>
                                    </div>
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      onClick={() => openInboxForMonitorSignal(monitorJobId, sample.item_id, 'simulated_policy_impact')}
                                    >
                                      Open in Inbox
                                    </Button>
                                  </div>
                                </div>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      ) : null}

                      {monitorJobId ? (
                        <div className="mt-4 border border-gray-200 rounded p-3 bg-white">
                          <div className="flex items-center justify-between gap-2 mb-3">
                            <div>
                              <div className="text-xs font-medium text-gray-700">Policy history</div>
                              <div className="text-[11px] text-gray-500">
                                {monitor.policy_history_count || 0} recorded change{(monitor.policy_history_count || 0) === 1 ? '' : 's'}
                              </div>
                            </div>
                          </div>
                          {(monitor.recent_policy_history || []).length === 0 ? (
                            <p className="text-xs text-gray-500">No recorded policy changes yet.</p>
                          ) : (
                            <div className="space-y-2">
                              {(monitor.recent_policy_history || []).map((entry, index) => {
                                const nextMode = canonicalReviewModeFromMonitorPolicyHistoryEntry(entry, 'next');
                                const prevMode = canonicalReviewModeFromMonitorPolicyHistoryEntry(entry, 'previous');
                                const nextAllowedRecommendations = canonicalAllowedRecommendationsFromMonitorPolicyHistoryEntry(entry, 'next');
                                const previousAllowedRecommendations = canonicalAllowedRecommendationsFromMonitorPolicyHistoryEntry(entry, 'previous');
                                const isCurrentEntry = index === 0;
                                const evaluationKey = `${monitorJobId}:${entry.id}`;
                                const evaluationDetail = healthPolicyEvaluations[evaluationKey];
                                return (
                                  <div key={entry.id} className="border border-gray-200 rounded p-3 bg-gray-50">
                                    <div className="flex items-start justify-between gap-3">
                                      <div>
                                        <div className="text-sm text-gray-900">
                                          {prevMode.replace(/_/g, ' ')} to <span className="font-medium">{nextMode.replace(/_/g, ' ')}</span>
                                        </div>
                                        <div className="text-[11px] text-gray-500 mt-1">
                                          {formatPolicyHistoryTimestamp(entry.at)}
                                          {entry.change_source ? ` via ${formatPolicyChangeSource(entry.change_source)}` : ''}
                                          {entry.actor_user_id ? ` by ${entry.actor_user_id}` : ''}
                                        </div>
                                        {entry.change_reason ? (
                                          <div className="text-[11px] text-gray-600 mt-1">{entry.change_reason}</div>
                                        ) : null}
                                        <div className="text-[11px] text-gray-500 mt-1">
                                          Allowlist: {nextAllowedRecommendations.join(', ') || 'None'}
                                        </div>
                                        {entry.evaluation_status ? (
                                          <div className="mt-2 flex flex-wrap gap-2">
                                            <span
                                              className={`text-[11px] px-2 py-1 rounded ${
                                                entry.evaluation_status === 'improving'
                                                  ? 'bg-emerald-100 text-emerald-700'
                                                  : entry.evaluation_status === 'degrading'
                                                    ? 'bg-rose-100 text-rose-700'
                                                    : entry.evaluation_status === 'mixed'
                                                      ? 'bg-amber-100 text-amber-800'
                                                      : 'bg-slate-100 text-slate-700'
                                              }`}
                                            >
                                              {formatPolicyEvaluationStatus(entry.evaluation_status)}
                                            </span>
                                            <span className="text-[11px] bg-white border border-gray-200 text-gray-700 px-2 py-1 rounded">
                                              Sample {entry.evaluation_sample_count}/{entry.evaluation_target_count || entry.evaluation_sample_count}
                                            </span>
                                            <span className="text-[11px] bg-white border border-gray-200 text-gray-700 px-2 py-1 rounded">
                                              Completed {formatSimulationCountDelta(entry.delta_counts?.follow_up_completed_count || 0)} · Failed {formatSimulationCountDelta(entry.delta_counts?.follow_up_failed_count || 0)} · Blocked {formatSimulationCountDelta(entry.delta_counts?.blocked_count || 0)}
                                            </span>
                                          </div>
                                        ) : null}
                                      </div>
                                      <div className="flex flex-col gap-2 shrink-0">
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          disabled={loadPolicyEvaluationMutation.isLoading}
                                          onClick={() =>
                                            loadPolicyEvaluationMutation.mutate({
                                              monitorJobId,
                                              historyEntryId: entry.id,
                                            })
                                          }
                                        >
                                          Compare before/after
                                        </Button>
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          disabled={rollbackMonitorPolicyMutation.isLoading}
                                          onClick={() =>
                                            simulateMonitorPolicyMutation.mutate({
                                              monitorJobId,
                                              data: {
                                                automation_profile: String(entry.previous_automation_profile || monitor.automation_profile || monitor.autonomy_mode || 'balanced'),
                                                automation_policy: {
                                                  follow_up_review_mode: prevMode,
                                                  allowed_recommendations: previousAllowedRecommendations,
                                                },
                                                mode: prevMode,
                                                allowed_recommendations: previousAllowedRecommendations,
                                                history_limit: 25,
                                              },
                                            })
                                          }
                                        >
                                          Preview restore
                                        </Button>
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          disabled={isCurrentEntry || rollbackMonitorPolicyMutation.isLoading}
                                          onClick={() =>
                                            rollbackMonitorPolicyMutation.mutate({
                                              monitorJobId,
                                              historyEntryId: entry.id,
                                            })
                                          }
                                        >
                                          Roll back
                                        </Button>
                                      </div>
                                    </div>
                                    {evaluationDetail ? (
                                      <div className="mt-3 border border-slate-200 rounded bg-white p-3">
                                        <div className="flex items-center gap-2 flex-wrap">
                                          <span
                                            className={`text-[11px] px-2 py-1 rounded ${
                                              evaluationDetail.evaluation_status === 'improving'
                                                ? 'bg-emerald-100 text-emerald-700'
                                                : evaluationDetail.evaluation_status === 'degrading'
                                                  ? 'bg-rose-100 text-rose-700'
                                                  : evaluationDetail.evaluation_status === 'mixed'
                                                    ? 'bg-amber-100 text-amber-800'
                                                    : 'bg-slate-100 text-slate-700'
                                            }`}
                                          >
                                            {formatPolicyEvaluationStatus(evaluationDetail.evaluation_status)}
                                          </span>
                                          <span className="text-[11px] text-slate-600">
                                            {evaluationDetail.evaluation_sample_count}/{evaluationDetail.evaluation_target_count} accepted signals after rollout
                                          </span>
                                        </div>
                                        <div className="grid grid-cols-3 gap-3 mt-3 text-[11px]">
                                          <div className="border border-slate-200 rounded p-2">
                                            <div className="font-medium text-slate-700">Before</div>
                                            <div className="mt-1 text-slate-600">
                                              Completed {evaluationDetail.before_counts.follow_up_completed_count} · Failed {evaluationDetail.before_counts.follow_up_failed_count} · Blocked {evaluationDetail.before_counts.blocked_count}
                                            </div>
                                          </div>
                                          <div className="border border-slate-200 rounded p-2">
                                            <div className="font-medium text-slate-700">After</div>
                                            <div className="mt-1 text-slate-600">
                                              Completed {evaluationDetail.after_counts.follow_up_completed_count} · Failed {evaluationDetail.after_counts.follow_up_failed_count} · Blocked {evaluationDetail.after_counts.blocked_count}
                                            </div>
                                          </div>
                                          <div className="border border-slate-200 rounded p-2">
                                            <div className="font-medium text-slate-700">Delta</div>
                                            <div className="mt-1 text-slate-600">
                                              Completed {formatSimulationCountDelta(evaluationDetail.delta_counts.follow_up_completed_count)} · Failed {formatSimulationCountDelta(evaluationDetail.delta_counts.follow_up_failed_count)} · Blocked {formatSimulationCountDelta(evaluationDetail.delta_counts.blocked_count)}
                                            </div>
                                          </div>
                                        </div>
                                        {(evaluationDetail.evaluation_reasons || []).length > 0 ? (
                                          <div className="mt-3 flex flex-wrap gap-2">
                                            {evaluationDetail.evaluation_reasons.map((reason) => (
                                              <span key={reason} className="text-[11px] bg-slate-50 text-slate-700 border border-slate-200 px-2 py-1 rounded">
                                                {reason}
                                              </span>
                                            ))}
                                          </div>
                                        ) : null}
                                        {(evaluationDetail.sample_items || []).length > 0 ? (
                                          <div className="mt-3 space-y-2">
                                            <div className="text-[11px] font-medium text-slate-700">Sample signals</div>
                                            {evaluationDetail.sample_items.map((sample) => (
                                              <div key={`${sample.period}-${sample.item_id}`} className="border border-slate-200 rounded p-2">
                                                <div className="flex items-start justify-between gap-3">
                                                  <div className="min-w-0">
                                                    <div className="text-xs font-medium text-slate-900">{sample.title}</div>
                                                    <div className="text-[11px] text-slate-600 mt-1">
                                                      {sample.period} · {sample.launch_status ? sample.launch_status.replace(/_/g, ' ') : 'no launch'}
                                                      {sample.outcome_status ? ` · ${sample.outcome_status.replace(/_/g, ' ')}` : ''}
                                                      {sample.recommendation_key ? ` · ${sample.recommendation_key}` : ''}
                                                    </div>
                                                    {sample.summary ? (
                                                      <div className="text-[11px] text-slate-600 mt-1">{sample.summary}</div>
                                                    ) : null}
                                                  </div>
                                                  <Button
                                                    size="sm"
                                                    variant="ghost"
                                                    onClick={() => openInboxForMonitorSignal(monitorJobId, sample.item_id, 'policy_evaluation_after_rollout')}
                                                  >
                                                    Open in Inbox
                                                  </Button>
                                                </div>
                                              </div>
                                            ))}
                                          </div>
                                        ) : null}
                                      </div>
                                    ) : null}
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      ) : null}

                      {(monitor.top_recommendations || []).length > 0 ? (
                        <div className="mt-4">
                          <div className="text-xs font-medium text-gray-700 mb-2">Top recommendation signals</div>
                          <div className="flex flex-wrap gap-2">
                            {monitor.top_recommendations.map((recommendation) => (
                              <span key={recommendation.recommendation_key} className="text-xs bg-slate-100 text-slate-700 px-2 py-1 rounded">
                                {recommendation.recommendation_key}: {recommendation.completed_count} complete / {recommendation.launch_count} launches
                              </span>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  )})}
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-4 overflow-y-auto">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h3 className="section-heading">Recommendation Performance</h3>
                      <p className="text-xs text-gray-500">Which bounded follow-ups are actually working.</p>
                    </div>
                  </div>
                  {filteredMonitorAnalytics.recommendations.length === 0 ? (
                    <p className="text-sm text-gray-500">No recommendation outcomes for the current filter set.</p>
                  ) : (
                    <div className="space-y-3">
                      {filteredMonitorAnalytics.recommendations.map((recommendation) => (
                        <div key={recommendation.recommendation_key} className="border border-gray-200 rounded p-3">
                          <div className="flex items-center justify-between gap-2">
                            <div className="font-medium text-gray-900">{recommendation.recommendation_key}</div>
                            <span
                              className={`text-xs px-2 py-1 rounded ${
                                recommendation.score_trend === 'positive'
                                  ? 'bg-emerald-100 text-emerald-700'
                                  : recommendation.score_trend === 'negative'
                                    ? 'bg-rose-100 text-rose-700'
                                    : 'bg-amber-100 text-amber-800'
                              }`}
                            >
                              {recommendation.score_trend}
                            </span>
                          </div>
                          <div className="text-xs text-gray-500 mt-2">
                            Success {recommendation.success_rate.toFixed(1)}% · Launches {recommendation.launch_count} · Monitors {recommendation.monitor_count}
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            Completed {recommendation.completed_count} · Failed {recommendation.failed_count} · Cancelled {recommendation.cancelled_count}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'fleet' && (
          <div className="w-full flex flex-col min-h-0 gap-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-1 bg-white border border-gray-200 rounded-lg p-4 space-y-3">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">Research Fleet</h2>
                  <p className="text-sm text-gray-500">Coordinate multiple domain profiles into a mostly automatic experiment portfolio.</p>
                </div>
                <input
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  placeholder="Portfolio title"
                  value={portfolioTitle}
                  onChange={(e) => setPortfolioTitle(e.target.value)}
                />
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={4}
                  placeholder="Portfolio objective"
                  value={portfolioObjective}
                  onChange={(e) => setPortfolioObjective(e.target.value)}
                />
                <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
                  <div className="text-xs font-medium text-gray-800 mb-2">Linked domain profiles</div>
                  <div className="space-y-2 max-h-56 overflow-auto">
                    {(((domainProfilesData as any)?.items || []) as DomainResearchProfile[]).map((profile) => (
                      <label key={profile.id} className="flex items-start gap-2 text-sm text-gray-700">
                        <input
                          type="checkbox"
                          checked={Boolean(portfolioProfileSelection[profile.id])}
                          onChange={(e) => setPortfolioProfileSelection((prev) => ({ ...prev, [profile.id]: e.target.checked }))}
                        />
                        <span>
                          <span className="font-medium text-gray-900">{profile.title}</span>
                          <span className="block text-xs text-gray-500">{profile.domain}</span>
                        </span>
                      </label>
                    ))}
                    {!(((domainProfilesData as any)?.items || []) as DomainResearchProfile[]).length ? (
                      <div className="text-xs text-gray-500">Create domain profiles first.</div>
                    ) : null}
                  </div>
                </div>
                {renderScientificSandboxManagementPanel()}
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={portfolioSandboxProfileId}
                  onChange={(e) => setPortfolioSandboxProfileId(e.target.value)}
                >
                  {portfolioAvailableSandboxProfiles.map((profile) => (
                    <option key={String(profile.id)} value={String(profile.id)}>
                      {String(profile.name)} ({String(profile.track_type || 'generic')})
                    </option>
                  ))}
                </select>
                <div className="flex gap-2">
                  <Button
                    variant="secondary"
                    disabled={createScientificResearchPackMutation.isLoading}
                    onClick={() => createScientificResearchPackMutation.mutate()}
                  >
                    {createScientificResearchPackMutation.isLoading ? 'Seeding pack...' : 'Seed Scientific Pack'}
                  </Button>
                  <Button
                    variant="primary"
                    disabled={createResearchPortfolioMutation.isLoading || !portfolioTitle.trim() || !portfolioObjective.trim() || selectedPortfolioProfileIds.length === 0}
                    onClick={() =>
                      createResearchPortfolioMutation.mutate({
                        title: portfolioTitle.trim(),
                        objective: portfolioObjective.trim(),
                        linked_profile_ids: selectedPortfolioProfileIds,
                        automation_profile: 'balanced',
                        automation_policy: {
                          ...DEFAULT_VALIDATION_POLICY,
                          duplicate_window_items: 120,
                        },
                        sandbox_profile_id: portfolioSandboxProfileId || resolveSandboxProfileId('compiler'),
                        start_immediately: true,
                      })
                    }
                  >
                    Start Fleet
                  </Button>
                  <Button variant="ghost" onClick={() => refetchResearchPortfolios()}>
                    <RefreshCw className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              <div className="col-span-2 bg-white border border-gray-200 rounded-lg p-4 min-h-0">
                {researchPortfoliosLoading ? (
                  <div className="flex justify-center items-center h-48"><LoadingSpinner /></div>
                ) : (
                  <div className="space-y-3">
                    {(((researchPortfoliosData as any)?.items || []) as ResearchPortfolio[]).map((portfolio) => {
                      const summary = (portfolio.latest_summary || {}) as Record<string, any>;
                      const autonomyMode = String(summary.autonomy_mode || portfolio.automation_profile || 'balanced');
                      const autonomySummary = (summary.autonomy_summary || {}) as Record<string, any>;
                      const effectivePolicy = ((portfolio.effective_policy || summary.effective_policy || portfolio.automation_policy || {}) as Record<string, any>) || {};
                      const policyDraft = portfolioPolicyDrafts[String(portfolio.id)] || buildResearchPortfolioPolicyDraft(portfolio);
                      const opportunities = Array.isArray(portfolio.opportunities) ? (portfolio.opportunities as ResearchOpportunity[]) : [];
                      const stageCounts = (summary.stage_counts || {}) as Record<string, any>;
                      const autonomyStateCounts = (summary.autonomy_state_counts || {}) as Record<string, any>;
                      const linkedProfiles = Array.isArray(portfolio.linked_profile_ids) ? portfolio.linked_profile_ids.length : 0;
                      const plansCount = Array.isArray(portfolio.latest_experiment_plan_ids) ? portfolio.latest_experiment_plan_ids.length : 0;
                      const validationCount = Array.isArray(portfolio.latest_validation_run_ids) ? portfolio.latest_validation_run_ids.length : 0;
                      const validationRuns = Array.isArray(portfolio.latest_validation_runs) ? portfolio.latest_validation_runs : [];
                      const recentValidationStats = validationRuns.reduce(
                        (acc, run) => {
                          const key = String(run.status || '').trim().toLowerCase();
                          if (!key) return acc;
                          acc[key] = Number(acc[key] || 0) + 1;
                          return acc;
                        },
                        {} as Record<string, number>
                      );
                      const childCount = Array.isArray(portfolio.child_job_ids) ? portfolio.child_job_ids.length : 0;
                      const queuedReviewsCount = Number(summary.queued_operator_reviews_count || 0);
                      const queuedReviewsByType = (summary.queued_operator_reviews_by_type || {}) as Record<string, any>;
                      const schedulerSummary = (summary.scheduler_summary || {}) as Record<string, any>;
                      const portfolioCardKey = buildAutonomyCardKey('fleet', String(portfolio.id));
                      const isPortfolioExpanded = Boolean(expandedPortfolioIds[String(portfolio.id)]);
                      return (
                        <div
                          key={portfolio.id}
                          ref={registerAutonomyCardRef(portfolioCardKey)}
                          className={`border border-gray-200 rounded-lg p-4 transition-colors ${highlightedAutonomyCardKey === portfolioCardKey ? AUTONOMY_FOCUS_CARD_CLASS : ''}`}
                        >
                          <div className="flex items-start justify-between gap-4">
                            <div className="min-w-0">
                              <div className="flex items-center gap-2 mb-1 flex-wrap">
                                <h3 className="section-heading">{portfolio.title}</h3>
                                <span className="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-700">{portfolio.status}</span>
                                <span className={`text-xs px-2 py-0.5 rounded ${autonomyMode === 'max_autonomy' ? 'bg-amber-100 text-amber-800' : 'bg-blue-100 text-blue-700'}`}>
                                  {autonomyMode === 'max_autonomy' ? 'max autonomy' : autonomyMode}
                                </span>
                                <span className="text-xs px-2 py-0.5 rounded bg-emerald-100 text-emerald-700">
                                  {opportunities.length} opportunities
                                </span>
                              </div>
                              <div className="text-sm text-gray-600 whitespace-pre-wrap">{portfolio.objective}</div>
                              <div className="text-xs text-gray-500 mt-2 flex flex-wrap gap-3">
                                <span>Profiles {linkedProfiles}</span>
                                <span>Plans {plansCount}</span>
                                <span>Validations {validationCount}</span>
                                <span>Follow-ups {childCount}</span>
                                {portfolio.last_run_at ? <span>Last run {new Date(portfolio.last_run_at).toLocaleString()}</span> : null}
                              </div>
                            </div>
                            <div className="flex gap-2 shrink-0 flex-wrap justify-end">
                              {['draft', 'completed', 'cancelled'].includes(portfolio.status) ? (
                                <Button size="sm" variant="primary" onClick={() => researchPortfolioActionMutation.mutate({ portfolioId: portfolio.id, action: 'start' })}>
                                  Start
                                </Button>
                              ) : null}
                              {portfolio.status === 'running' ? (
                                <Button size="sm" variant="secondary" onClick={() => researchPortfolioActionMutation.mutate({ portfolioId: portfolio.id, action: 'pause' })}>
                                  Pause
                                </Button>
                              ) : null}
                              {portfolio.status === 'paused' ? (
                                <Button size="sm" variant="secondary" onClick={() => researchPortfolioActionMutation.mutate({ portfolioId: portfolio.id, action: 'resume' })}>
                                  Resume
                                </Button>
                              ) : null}
                              <Button size="sm" variant="ghost" onClick={() => researchPortfolioActionMutation.mutate({ portfolioId: portfolio.id, action: 'run_now' })}>
                                Run Now
                              </Button>
                            </div>
                          </div>
                          <details
                            className="mt-3 bg-gray-50 border border-gray-100 rounded-lg p-3"
                            open={isPortfolioExpanded}
                            onToggle={(e) => {
                              const nextOpen = (e.currentTarget as HTMLDetailsElement).open;
                              setExpandedPortfolioIds((prev) => ({ ...prev, [String(portfolio.id)]: nextOpen }));
                            }}
                          >
                            <summary className="cursor-pointer text-xs font-medium text-gray-800">Portfolio state</summary>
                            <div className="mt-3 space-y-3 text-xs text-gray-700">
                              <SharedAutonomyMetricGrid
                                columns="grid-cols-5"
                                items={[
                                  { label: 'Discovered', value: Number(stageCounts.discovered || 0) },
                                  { label: 'Planned', value: Number(stageCounts.planned || 0) },
                                  { label: 'Validating', value: Number(stageCounts.validating || 0) },
                                  { label: 'Validation runs', value: validationCount, detail: `Run ${Number(recentValidationStats.running || 0)} · Blocked ${Number(recentValidationStats.blocked || 0)}` },
                                  { label: 'Suppressed', value: Number(stageCounts.suppressed || 0) },
                                ]}
                              />
                              <SharedAutonomyMetricGrid
                                items={[
                                  { label: 'Blocked', value: Number(autonomySummary.blocked_opportunities_count || 0) },
                                  { label: 'Dupes suppressed', value: Number(autonomySummary.suppressed_duplicates_count || 0) },
                                  { label: 'Plans launched', value: Number(autonomySummary.created_experiment_plan_count || 0) },
                                  { label: 'Follow-ups launched', value: Number(autonomySummary.launched_follow_up_job_count || 0) },
                                ]}
                              />
                              <SharedAutonomyMetricGrid
                                items={[
                                  { label: 'Eligible now', value: Number(autonomyStateCounts.eligible || 0) },
                                  { label: 'Cooling down', value: Number(autonomyStateCounts.cooldown || 0) },
                                  { label: 'Waiting on change', value: Number(autonomyStateCounts.completed_waiting_change || 0) },
                                  { label: 'Structurally blocked', value: Number(autonomyStateCounts.blocked_structural || 0) },
                                ]}
                              />
                              <SharedAutonomyMetricGrid
                                items={[
                                  { label: 'Queued reviews', value: queuedReviewsCount },
                                  { label: 'Follow-up approvals', value: Number(queuedReviewsByType.follow_up_recommendation || 0) },
                                  { label: 'Policy reviews', value: Number(queuedReviewsByType.policy_review || 0) },
                                  { label: 'Budget reviews', value: Number(queuedReviewsByType.budget_review || 0) },
                                ]}
                              />
                              <SharedAutonomyMetricGrid
                                items={[
                                  { label: 'Next run', value: schedulerSummary.next_run_at ? new Date(String(schedulerSummary.next_run_at)).toLocaleString() : 'n/a' },
                                  { label: 'Pending approvals', value: Number(schedulerSummary.pending_follow_up_approvals_count || 0) },
                                  { label: 'Manual recommendations', value: Number(schedulerSummary.manual_follow_up_recommendations_count || 0) },
                                  { label: 'Suppressed relaunches', value: Number(schedulerSummary.suppressed_relaunches_count || 0) },
                                ]}
                              />
                              <SharedPortfolioLikeAutonomyControls
                                draft={policyDraft}
                                applyLabel="Apply settings"
                                disabled={updateResearchPortfolioMutation.isLoading}
                                onApply={() => submitPortfolioPolicyDraft(portfolio)}
                                onFieldChange={(field, value) => updatePortfolioPolicyDraftField(portfolio, field, value)}
                              />
                              <div className="text-gray-500">
                                Effective policy: confidence {Number(effectivePolicy.confidence_threshold || 0).toFixed(2)}
                                {' '}· readiness {Number(effectivePolicy.experiment_readiness_threshold || 0).toFixed(2)}
                                {' '}· validation {effectivePolicy.auto_launch_experiment_runs ? 'on' : 'off'}
                                {' '}· review {formatReviewModeLabel(effectivePolicy.follow_up_review_mode || 'auto_launch_safe')}
                              </div>
                              {renderBulkFollowUpControls(
                                'fleet',
                                String(portfolio.id),
                                summary.pending_follow_up_approvals as Array<Record<string, any>> | undefined,
                                summary.manual_follow_up_recommendations as Array<Record<string, any>> | undefined,
                                summary.suppressed_relaunches as Array<Record<string, any>> | undefined,
                                opportunities as Array<Record<string, any>> | undefined,
                              )}
                              <SharedAutonomyReviewLists
                                sections={[
                                  { title: 'Queued operator reviews', rows: summary.queued_operator_reviews as Array<Record<string, any>> | undefined },
                                  {
                                    title: 'Pending follow-up approvals',
                                    rows: summary.pending_follow_up_approvals as Array<Record<string, any>> | undefined,
                                    renderRow: (row, idx) => renderInlineFollowUpApprovalRow('fleet', String(portfolio.id), row, idx),
                                  },
                                  {
                                    title: 'Manual follow-up recommendations',
                                    rows: summary.manual_follow_up_recommendations as Array<Record<string, any>> | undefined,
                                    renderRow: (row, idx) => renderInlineManualRecommendationRow(
                                      'fleet',
                                      String(portfolio.id),
                                      row,
                                      idx,
                                      opportunities as Array<Record<string, any>> | undefined,
                                    ),
                                  },
                                  {
                                    title: 'Suppressed relaunches',
                                    rows: summary.suppressed_relaunches as Array<Record<string, any>> | undefined,
                                    renderRow: (row, idx) => renderInlineSuppressedRelaunchRow(
                                      'fleet',
                                      String(portfolio.id),
                                      row,
                                      idx,
                                      opportunities as Array<Record<string, any>> | undefined,
                                    ),
                                  },
                                ]}
                              />
                              {Array.isArray(summary.auto_launch_decisions) && summary.auto_launch_decisions.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Automatic actions</div>
                                  <div className="mt-1 space-y-1">
                                    {summary.auto_launch_decisions.slice(0, 6).map((row: Record<string, any>, idx: number) => (
                                      <div key={`${String(row.type || 'action')}-${idx}`}>
                                        {String(row.type || 'action').replace(/_/g, ' ')}
                                        {row.plan_id ? ` · Plan ${String(row.plan_id)}` : ''}
                                        {row.job_id ? ` · Job ${String(row.job_id)}` : ''}
                                        {row.reason_code ? ` · ${String(row.reason_code)}` : ''}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              ) : null}
                              {Array.isArray(summary.blocked_opportunities) && summary.blocked_opportunities.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Blocked opportunities</div>
                                  <div className="mt-1 space-y-1">
                                    {summary.blocked_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                      return renderAutonomySummaryRow(
                                        'fleet',
                                        String(portfolio.id),
                                        'suppressed',
                                        row,
                                        idx,
                                        <>
                                          <div>{String(row.title || row.canonical_key || 'Blocked opportunity')}{row.last_blocked_reason_code ? ` · ${String(row.last_blocked_reason_code)}` : ''}</div>
                                          {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('fleet', String(portfolio.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'fleet', ownerId: String(portfolio.id) })}
                                        </>
                                      );
                                    })}
                                  </div>
                                </div>
                              ) : null}
                              {Array.isArray(summary.completed_waiting_change_opportunities) && summary.completed_waiting_change_opportunities.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Waiting on evidence change</div>
                                  <div className="mt-1 space-y-1">
                                    {summary.completed_waiting_change_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                      return renderAutonomySummaryRow(
                                        'fleet',
                                        String(portfolio.id),
                                        'suppressed',
                                        row,
                                        idx,
                                        <>
                                          <div>{String(row.title || row.canonical_key || 'Completed opportunity')}{row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                          {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('fleet', String(portfolio.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'fleet', ownerId: String(portfolio.id) })}
                                        </>
                                      );
                                    })}
                                  </div>
                                </div>
                              ) : null}
                              {Array.isArray(summary.cooldown_opportunities) && summary.cooldown_opportunities.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Cooldown opportunities</div>
                                  <div className="mt-1 space-y-1">
                                    {summary.cooldown_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                      return renderAutonomySummaryRow(
                                        'fleet',
                                        String(portfolio.id),
                                        'suppressed',
                                        row,
                                        idx,
                                        <>
                                          <div>{String(row.title || row.canonical_key || 'Cooldown opportunity')}{row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                          {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('fleet', String(portfolio.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'fleet', ownerId: String(portfolio.id) })}
                                        </>
                                      );
                                    })}
                                  </div>
                                </div>
                              ) : null}
                              {Array.isArray(summary.skipped_opportunities) && summary.skipped_opportunities.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Skipped opportunities</div>
                                  <div className="mt-1 space-y-1">
                                    {summary.skipped_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                      return renderAutonomySummaryRow(
                                        'fleet',
                                        String(portfolio.id),
                                        'suppressed',
                                        row,
                                        idx,
                                        <>
                                          <div>{String(row.title || row.canonical_key || 'Skipped opportunity')}{row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                          {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('fleet', String(portfolio.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'fleet', ownerId: String(portfolio.id) })}
                                        </>
                                      );
                                    })}
                                  </div>
                                </div>
                              ) : null}
                              {validationRuns.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Recent validation runs</div>
                                  <div className="mt-2">{renderScientificValidationRuns(validationRuns as any)}</div>
                                </div>
                              ) : null}
                              {opportunities.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Top opportunities</div>
                                  <div className="mt-2 space-y-2">
                                    {opportunities.slice(0, 6).map((row) => {
                                      const opportunityRowKey = buildAutonomyOpportunityRowKey('fleet', String(portfolio.id), String(row.opportunity_id || row.canonical_key || row.title));
                                      const opportunityNoteId = String((Array.isArray(row.source_note_ids) && row.source_note_ids.length > 0
                                        ? row.source_note_ids[0]
                                        : (Array.isArray(portfolio.latest_note_ids) && portfolio.latest_note_ids.length > 0 ? portfolio.latest_note_ids[0] : '')) || '').trim();
                                      return (
                                      <div
                                        key={String(row.opportunity_id || row.canonical_key || row.title)}
                                        ref={registerAutonomyRowRef(opportunityRowKey)}
                                        className={`border border-gray-100 rounded p-2 transition-colors ${highlightedAutonomyRowKey === opportunityRowKey ? AUTONOMY_FOCUS_ROW_CLASS : ''}`}
                                      >
                                        <div className="flex items-center justify-between gap-2">
                                          <div className="font-medium text-gray-900">{String(row.title || row.canonical_key)}</div>
                                          <span className={`text-[11px] px-2 py-0.5 rounded ${researchOpportunityStageClass(row.stage)}`}>
                                            {String(row.stage || 'discovered')}
                                          </span>
                                        </div>
                                        <div className="mt-1 text-gray-500">
                                          Confidence {Number(row.confidence || 0).toFixed(2)}
                                          {' '}· Novelty {Number(row.novelty || 0).toFixed(2)}
                                          {' '}· Readiness {Number(row.readiness || 0).toFixed(2)}
                                        </div>
                                        {row.operator_note ? (
                                          <div className="mt-1 text-gray-500">Note: {row.operator_note}</div>
                                        ) : null}
                                        <div className="mt-2 text-gray-500">
                                          Plans {Array.isArray(row.linked_experiment_plan_ids) ? row.linked_experiment_plan_ids.length : 0}
                                          {' '}· Runs {Array.isArray(row.linked_validation_run_ids) ? row.linked_validation_run_ids.length : 0}
                                          {' '}· Jobs {Array.isArray(row.child_job_ids) ? row.child_job_ids.length : 0}
                                        </div>
                                        {String(row.latest_experiment_plan_id || row.latest_validation_run_id || row.latest_validation_job_id || '').trim() ? (
                                          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-gray-500">
                                            {row.latest_experiment_plan_id ? <span>Latest plan {String(row.latest_experiment_plan_id).slice(0, 8)}</span> : null}
                                            {row.latest_validation_run_id ? <span>Run {String(row.latest_validation_run_id).slice(0, 8)}</span> : null}
                                            {row.latest_validation_status ? <span>Status {String(row.latest_validation_status).replace(/_/g, ' ')}</span> : null}
                                            {row.latest_validation_blocked_reason_code ? <span>Blocked {String(row.latest_validation_blocked_reason_code).replace(/_/g, ' ')}</span> : null}
                                            {row.latest_experiment_plan_id && opportunityNoteId ? (
                                              <Button
                                                size="sm"
                                                variant="ghost"
                                                className="!px-2 !py-1 !h-auto text-xs"
                                                onClick={() => navigate(buildResearchNoteExperimentUrl(opportunityNoteId, { plan: String(row.latest_experiment_plan_id) }))}
                                              >
                                                Open plan
                                              </Button>
                                            ) : null}
                                            {row.latest_validation_run_id && opportunityNoteId ? (
                                              <Button
                                                size="sm"
                                                variant="ghost"
                                                className="!px-2 !py-1 !h-auto text-xs"
                                                onClick={() => navigate(buildResearchNoteExperimentUrl(opportunityNoteId, { run: String(row.latest_validation_run_id) }))}
                                              >
                                                Open run
                                              </Button>
                                            ) : null}
                                            {row.latest_validation_job_id ? (
                                              <Button
                                                size="sm"
                                                variant="ghost"
                                                className="!px-2 !py-1 !h-auto text-xs"
                                                onClick={() => navigate(buildAutonomousAgentsUrl(String(row.latest_validation_job_id)), { replace: true })}
                                              >
                                                Open validation job
                                              </Button>
                                            ) : null}
                                          </div>
                                        ) : null}
                                        <div className="mt-1 text-gray-500">
                                          Autonomy {String(row.autonomy_state || 'eligible').replace(/_/g, ' ')}
                                          {row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : ''}
                                          {row.next_eligible_at ? ` · Next eligible ${new Date(row.next_eligible_at).toLocaleString()}` : ''}
                                        </div>
                                        {renderOpportunityReevaluationReviewMeta(row, (url) => navigate(url))}
                                        {renderOpportunityFollowUpOutcomeMeta(row)}
                                        {renderOpportunityExplainabilityPanel(opportunityRowKey, row, { surface: 'fleet', ownerId: String(portfolio.id) })}
                                        <div className="mt-2 flex flex-wrap gap-2">
                                          {row.decision_state !== 'accepted' ? (
                                            <Button
                                              size="sm"
                                              variant="secondary"
                                              onClick={() => researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'accept' })}
                                            >
                                              Accept
                                            </Button>
                                          ) : null}
                                          {row.decision_state !== 'suppressed' ? (
                                            <Button
                                              size="sm"
                                              variant="ghost"
                                              onClick={() => beginOpportunitySuppression('fleet', portfolio.id, row)}
                                            >
                                              Suppress
                                            </Button>
                                          ) : (
                                            <Button
                                              size="sm"
                                              variant="ghost"
                                              onClick={() => researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'reopen' })}
                                            >
                                              Reopen
                                            </Button>
                                          )}
                                          {row.decision_state === 'accepted' ? (
                                            <Button
                                              size="sm"
                                              variant="primary"
                                              onClick={() => researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'materialize_experiment', startImmediately: true })}
                                            >
                                              Run Experiment
                                            </Button>
                                          ) : null}
                                          <Button
                                            size="sm"
                                            variant="ghost"
                                            disabled={Array.isArray(row.linked_experiment_plan_ids) && row.linked_experiment_plan_ids.length > 0}
                                            onClick={() => researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'create_plan' })}
                                          >
                                            Create Plan
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="ghost"
                                            disabled={Array.isArray(row.linked_validation_run_ids) && row.linked_validation_run_ids.length > 0}
                                            onClick={() => researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'launch_validation' })}
                                          >
                                            Launch Validation
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="ghost"
                                            disabled={canRelaunchOpportunityRow(row) ? false : Array.isArray(row.child_job_ids) && row.child_job_ids.length > 0}
                                            onClick={() => (
                                              canRelaunchOpportunityRow(row)
                                                ? beginOpportunityRelaunch('fleet', String(portfolio.id), row)
                                                : researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'launch_follow_up' })
                                            )}
                                          >
                                            {canRelaunchOpportunityRow(row) ? 'Relaunch Follow-up' : 'Follow-up'}
                                          </Button>
                                        </div>
                                        {opportunityNoteDraft?.surface === 'fleet'
                                        && String(opportunityNoteDraft.ownerId) === String(portfolio.id)
                                        && String(opportunityNoteDraft.opportunityId) === String(row.opportunity_id) ? (
                                          <div className={`mt-2 rounded p-2 ${opportunityNoteDraft.mode === 'suppress' ? 'border border-rose-200 bg-rose-50' : 'border border-emerald-200 bg-emerald-50'}`}>
                                            <div className={`text-[11px] font-medium ${opportunityNoteDraft.mode === 'suppress' ? 'text-rose-700' : 'text-emerald-700'}`}>
                                              {opportunityNoteDraft.mode === 'suppress' ? 'Suppression note' : 'Relaunch note'}
                                            </div>
                                            <textarea
                                              aria-label={opportunityNoteDraft.mode === 'suppress' ? 'Fleet suppression note' : 'Fleet relaunch note'}
                                              className={`mt-2 w-full rounded px-2 py-1 text-xs ${opportunityNoteDraft.mode === 'suppress' ? 'border border-rose-200' : 'border border-emerald-200'}`}
                                              rows={3}
                                              value={opportunityNoteDraft.value}
                                              onChange={(e) => setOpportunityNoteDraft((prev) => prev ? { ...prev, value: e.target.value } : prev)}
                                            />
                                            <div className="mt-2 flex gap-2">
                                              <Button size="sm" variant="secondary" onClick={submitOpportunityAction}>
                                                {opportunityNoteDraft.mode === 'suppress' ? 'Save suppression' : 'Relaunch follow-up'}
                                              </Button>
                                              <Button size="sm" variant="ghost" onClick={cancelOpportunityAction}>
                                                Cancel
                                              </Button>
                                            </div>
                                          </div>
                                        ) : null}
                                      </div>
                                    );})}
                                  </div>
                                </div>
                              ) : null}
                            </div>
                          </details>
                        </div>
                      );
                    })}
                    {!(((researchPortfoliosData as any)?.items || []) as ResearchPortfolio[]).length ? (
                      <div className="text-sm text-gray-500">No research portfolios yet.</div>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'domain' && (
          <div className="w-full flex flex-col min-h-0 gap-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-1 bg-white border border-gray-200 rounded-lg p-4 space-y-3">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">Domain Profiles</h2>
                  <p className="text-sm text-gray-500">Saved R&D research monitors that persist notes, delta summaries, and experiment plans.</p>
                </div>
                <Button
                  variant="secondary"
                  disabled={createScientificResearchPackMutation.isLoading}
                  onClick={() => createScientificResearchPackMutation.mutate()}
                >
                  {createScientificResearchPackMutation.isLoading ? 'Seeding scientific pack...' : 'Seed Compiler + Microarch Pack'}
                </Button>
                {renderScientificSandboxManagementPanel()}
                <input
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  placeholder="Profile title"
                  value={domainProfileTitle}
                  onChange={(e) => setDomainProfileTitle(e.target.value)}
                />
                <input
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  placeholder="Domain or topic"
                  value={domainProfileTopic}
                  onChange={(e) => setDomainProfileTopic(e.target.value)}
                />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <select
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={domainProfileTrackType}
                    onChange={(e) => setDomainProfileTrackType(e.target.value as any)}
                  >
                    {DOMAIN_TRACK_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>{option.label} track</option>
                    ))}
                  </select>
                  <select
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={domainProfileSourceScope}
                    onChange={(e) => setDomainProfileSourceScope(e.target.value as any)}
                  >
                    {DOMAIN_SOURCE_SCOPE_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                </div>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={domainProfileSandboxProfileId}
                  onChange={(e) => setDomainProfileSandboxProfileId(e.target.value)}
                >
                  {domainAvailableSandboxProfiles.map((profile) => (
                    <option key={String(profile.id)} value={String(profile.id)}>
                      {String(profile.name)} ({String(profile.track_type || 'generic')})
                    </option>
                  ))}
                </select>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={4}
                  placeholder="Research objective"
                  value={domainProfileObjective}
                  onChange={(e) => setDomainProfileObjective(e.target.value)}
                />
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={3}
                  placeholder="Monitor queries, one per line"
                  value={domainProfileQueriesText}
                  onChange={(e) => setDomainProfileQueriesText(e.target.value)}
                />
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={3}
                  placeholder="Benchmark queries, one per line"
                  value={domainProfileBenchmarkQueriesText}
                  onChange={(e) => setDomainProfileBenchmarkQueriesText(e.target.value)}
                />
                {codeSources.length > 0 ? (
                  <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
                    <div className="text-xs font-medium text-gray-800 mb-2">Repository evidence sources</div>
                    <div className="space-y-2 max-h-36 overflow-auto">
                      {codeSources.map((source: any) => (
                        <label key={String(source.id)} className="flex items-start gap-2 text-sm text-gray-700">
                          <input
                            type="checkbox"
                            checked={Boolean(domainProfileRepoSelection[String(source.id)])}
                            onChange={(e) => setDomainProfileRepoSelection((prev) => ({ ...prev, [String(source.id)]: e.target.checked }))}
                          />
                          <span>
                            <span className="font-medium text-gray-900">{String(source.name || source.id)}</span>
                            <span className="block text-xs text-gray-500">{String(source.source_type || '').toLowerCase()}</span>
                          </span>
                        </label>
                      ))}
                    </div>
                  </div>
                ) : null}
                <input
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  placeholder="Cadence in minutes"
                  value={domainProfileCadenceMinutes}
                  onChange={(e) => setDomainProfileCadenceMinutes(e.target.value)}
                />
                <div className="flex gap-2">
                  <Button
                    variant="primary"
                    disabled={createDomainProfileMutation.isLoading || !domainProfileTitle.trim() || !domainProfileTopic.trim() || !domainProfileObjective.trim()}
                    onClick={() =>
                      createDomainProfileMutation.mutate({
                        title: domainProfileTitle.trim(),
                        domain: domainProfileTopic.trim(),
                        objective: domainProfileObjective.trim(),
                        track_type: domainProfileTrackType,
                        source_scope: domainProfileSourceScope,
                        research_mode: 'literature_to_hypothesis',
                        monitor_queries: splitUniqueLines(domainProfileQueriesText, 12),
                        repo_source_ids: selectedDomainProfileRepoSourceIds.length ? selectedDomainProfileRepoSourceIds : undefined,
                        benchmark_queries: splitUniqueLines(domainProfileBenchmarkQueriesText, 16),
                        sandbox_profile_id: domainProfileSandboxProfileId || resolveSandboxProfileId(domainProfileTrackType),
                        scoring_policy: {
                          minimum_subscore: 0.6,
                          minimum_supporting_sources: 2,
                          weights: { novelty: 0.4, evidence: 0.35, testability: 0.25 },
                        },
                        selection_policy: { max_candidates: 10, max_hypotheses: 3 },
                        automation_profile: 'balanced',
                        automation_policy: DEFAULT_VALIDATION_POLICY,
                        interval_minutes: Number(domainProfileCadenceMinutes) > 0 ? Number(domainProfileCadenceMinutes) : 1440,
                        persist_artifacts: true,
                        auto_launch_follow_up: true,
                        auto_create_experiment_plans: true,
                        start_immediately: true,
                      })
                    }
                  >
                    Start Monitor
                  </Button>
                  <Button variant="ghost" onClick={() => refetchDomainProfiles()}>
                    <RefreshCw className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              <div className="col-span-2 bg-white border border-gray-200 rounded-lg p-4 min-h-0">
                {domainProfilesLoading ? (
                  <div className="flex justify-center items-center h-48"><LoadingSpinner /></div>
                ) : (
                  <div className="space-y-3">
                    {(((domainProfilesData as any)?.items || []) as DomainResearchProfile[]).map((profile) => {
                      const summary = (profile.latest_summary || {}) as Record<string, any>;
                      const ideaTitles = Array.isArray(summary.ranked_opportunities) ? summary.ranked_opportunities.slice(0, 3) : [];
                      const opportunities = Array.isArray(profile.opportunities) ? profile.opportunities : [];
                      const autonomyMode = String(summary.autonomy_mode || profile.automation_profile || 'balanced');
                      const effectivePolicy = ((profile.effective_policy || summary.effective_policy || profile.automation_policy || {}) as Record<string, any>) || {};
                      const autonomyStateCounts = (summary.autonomy_state_counts || {}) as Record<string, any>;
                      const schedulerSummary = (summary.scheduler_summary || {}) as Record<string, any>;
                      const queuedReviewsCount = Number(summary.queued_operator_reviews_count || 0);
                      const policyDraft = domainProfilePolicyDrafts[String(profile.id)] || buildDomainResearchProfilePolicyDraft(profile);
                      const notesCount = Array.isArray(profile.latest_note_ids) ? profile.latest_note_ids.length : 0;
                      const plansCount = Array.isArray(profile.latest_experiment_plan_ids) ? profile.latest_experiment_plan_ids.length : 0;
                      const validationRuns = Array.isArray(profile.latest_validation_runs) ? profile.latest_validation_runs : [];
                      const delta = (summary.delta_since_last_run || {}) as Record<string, any>;
                      const profileCardKey = buildAutonomyCardKey('domain', String(profile.id));
                      const isProfileExpanded = Boolean(expandedDomainProfileIds[String(profile.id)]);
                      return (
                        <div
                          key={profile.id}
                          ref={registerAutonomyCardRef(profileCardKey)}
                          className={`border border-gray-200 rounded-lg p-4 transition-colors ${highlightedAutonomyCardKey === profileCardKey ? AUTONOMY_FOCUS_CARD_CLASS : ''}`}
                        >
                          <div className="flex items-start justify-between gap-4">
                            <div className="min-w-0">
                              <div className="flex items-center gap-2 mb-1 flex-wrap">
                                <h3 className="section-heading">{profile.title}</h3>
                                <span className="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-700">{profile.status}</span>
                                <span className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-700">{profile.domain}</span>
                                <span className="text-xs px-2 py-0.5 rounded bg-indigo-100 text-indigo-700">
                                  {String(profile.track_type || 'generic').replaceAll('_', ' ')}
                                </span>
                                {plansCount > 0 ? (
                                  <span className="text-xs px-2 py-0.5 rounded bg-emerald-100 text-emerald-700">
                                    {plansCount} experiment plan{plansCount === 1 ? '' : 's'}
                                  </span>
                                ) : null}
                              </div>
                              <div className="text-sm text-gray-600 whitespace-pre-wrap">{profile.objective}</div>
                              <div className="text-xs text-gray-500 mt-2 flex flex-wrap gap-3">
                                <span>Cadence {profile.interval_minutes}m</span>
                                <span>Mode {String(profile.research_mode || 'literature_to_hypothesis').replaceAll('_', ' ')}</span>
                                <span>Scope {String(profile.source_scope || 'kb_plus_arxiv').replaceAll('_', ' ')}</span>
                                <span>Notes {notesCount}</span>
                                <span>Plans {plansCount}</span>
                                <span>Validations {Array.isArray(profile.latest_validation_run_ids) ? profile.latest_validation_run_ids.length : 0}</span>
                                {profile.last_run_at ? <span>Last run {new Date(profile.last_run_at).toLocaleString()}</span> : null}
                              </div>
                              {summary.domain_summary ? (
                                <div className="mt-2 text-xs text-gray-600">{String(summary.domain_summary)}</div>
                              ) : null}
                              {Number(delta.new_signal_count || 0) > 0 ? (
                                <div className="mt-2 text-xs text-emerald-700">
                                  New signals {Number(delta.new_signal_count || 0)}
                                  {Array.isArray(delta.new_idea_titles) && delta.new_idea_titles.length > 0 ? ` · ${delta.new_idea_titles.slice(0, 2).join(', ')}` : ''}
                                </div>
                              ) : null}
                            </div>
                            <div className="flex gap-2 shrink-0 flex-wrap justify-end">
                              {['draft', 'completed', 'cancelled'].includes(profile.status) ? (
                                <Button size="sm" variant="primary" onClick={() => domainProfileActionMutation.mutate({ profileId: profile.id, action: 'start' })}>
                                  Start
                                </Button>
                              ) : null}
                              {profile.status === 'running' ? (
                                <Button size="sm" variant="secondary" onClick={() => domainProfileActionMutation.mutate({ profileId: profile.id, action: 'pause' })}>
                                  Pause
                                </Button>
                              ) : null}
                              {profile.status === 'paused' ? (
                                <Button size="sm" variant="secondary" onClick={() => domainProfileActionMutation.mutate({ profileId: profile.id, action: 'resume' })}>
                                  Resume
                                </Button>
                              ) : null}
                              <Button size="sm" variant="ghost" onClick={() => domainProfileActionMutation.mutate({ profileId: profile.id, action: 'run_now' })}>
                                Run Now
                              </Button>
                              {profile.status !== 'cancelled' ? (
                                <Button size="sm" variant="ghost" onClick={() => domainProfileActionMutation.mutate({ profileId: profile.id, action: 'cancel' })}>
                                  Cancel
                                </Button>
                              ) : null}
                            </div>
                          </div>
                          <details
                            className="mt-3 bg-gray-50 border border-gray-100 rounded-lg p-3"
                            open={isProfileExpanded}
                            onToggle={(e) => {
                              const nextOpen = (e.currentTarget as HTMLDetailsElement).open;
                              setExpandedDomainProfileIds((prev) => ({ ...prev, [String(profile.id)]: nextOpen }));
                            }}
                          >
                            <summary className="cursor-pointer text-xs font-medium text-gray-800">Latest research ops state</summary>
                            <div className="mt-3 space-y-3 text-xs text-gray-700">
                              <SharedAutonomyMetricGrid
                                items={[
                                  {
                                    label: 'Fresh evidence',
                                    value: `Docs ${Array.isArray(delta.new_document_ids) ? delta.new_document_ids.length : 0} · Repo ${Array.isArray(delta.new_repo_document_ids) ? delta.new_repo_document_ids.length : 0} · Papers ${Array.isArray(delta.new_paper_ids) ? delta.new_paper_ids.length : 0}`,
                                  },
                                  {
                                    label: 'Novel ideas',
                                    value: `${Number((summary.novelty_summary || {}).new_idea_count || 0)} new`,
                                    detail: `Repeated ${Number((summary.novelty_summary || {}).repeated_idea_count || 0)}`,
                                  },
                                  {
                                    label: 'Automation',
                                    value: `${formatAutonomyLabel(autonomyMode)} · review ${formatReviewModeLabel(effectivePolicy.follow_up_review_mode || 'auto_launch_safe')}`,
                                    detail: `Confidence ${Number(effectivePolicy.confidence_threshold ?? profile.confidence_threshold ?? 0.7).toFixed(2)} · Sandbox ${String(scientificSandboxProfileById[String(profile.sandbox_profile_id || '')]?.name || profile.sandbox_profile_id || 'default')}`,
                                  },
                                  {
                                    label: 'Autonomy state',
                                    value: `Eligible ${Number(autonomyStateCounts.eligible || 0)} · Active ${Number(autonomyStateCounts.active || 0)}`,
                                    detail: `Waiting change ${Number(autonomyStateCounts.completed_waiting_change || 0)} · Structural blocked ${Number(autonomyStateCounts.blocked_structural || 0)}`,
                                  },
                                ]}
                              />
                              <SharedAutonomyMetricGrid
                                items={[
                                  { label: 'Next run', value: schedulerSummary.next_run_at ? new Date(String(schedulerSummary.next_run_at)).toLocaleString() : 'Not scheduled' },
                                  { label: 'Pending approvals', value: Number(schedulerSummary.pending_follow_up_approvals_count || 0) },
                                  { label: 'Manual recommendations', value: Number(schedulerSummary.manual_follow_up_recommendations_count || 0) },
                                  { label: 'Suppressed relaunches', value: Number(schedulerSummary.suppressed_relaunches_count || 0) },
                                ]}
                              />
                              <SharedPortfolioLikeAutonomyControls
                                draft={policyDraft}
                                applyLabel="Save"
                                disabled={updateDomainProfileMutation.isLoading}
                                onApply={() => submitDomainProfilePolicyDraft(profile)}
                                onFieldChange={(field, value) => updateDomainProfilePolicyDraftField(profile, field, value)}
                              />
                              <div className="text-[11px] text-gray-500">
                                Queued reviews {queuedReviewsCount}
                              </div>
                              {renderBulkFollowUpControls(
                                'domain',
                                String(profile.id),
                                summary.pending_follow_up_approvals as Array<Record<string, any>> | undefined,
                                summary.manual_follow_up_recommendations as Array<Record<string, any>> | undefined,
                                summary.suppressed_relaunches as Array<Record<string, any>> | undefined,
                                opportunities as Array<Record<string, any>> | undefined,
                              )}
                              {Array.isArray(summary.blocked_opportunities) && summary.blocked_opportunities.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Blocked opportunities</div>
                                  <div className="mt-1 space-y-1">
                                    {summary.blocked_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                      return renderAutonomySummaryRow(
                                        'domain',
                                        String(profile.id),
                                        'suppressed',
                                        row,
                                        idx,
                                        <>
                                          <div>{String(row.title || row.canonical_key || 'Blocked opportunity')}{row.last_blocked_reason_code ? ` · ${String(row.last_blocked_reason_code)}` : ''}</div>
                                          {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('domain', String(profile.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'domain', ownerId: String(profile.id) })}
                                        </>
                                      );
                                    })}
                                  </div>
                                </div>
                              ) : null}
                              {Array.isArray(summary.completed_waiting_change_opportunities) && summary.completed_waiting_change_opportunities.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Waiting on evidence change</div>
                                  <div className="mt-1 space-y-1">
                                    {summary.completed_waiting_change_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                      return renderAutonomySummaryRow(
                                        'domain',
                                        String(profile.id),
                                        'suppressed',
                                        row,
                                        idx,
                                        <>
                                          <div>{String(row.title || row.canonical_key || 'Completed opportunity')}{row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                          {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('domain', String(profile.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'domain', ownerId: String(profile.id) })}
                                        </>
                                      );
                                    })}
                                  </div>
                                </div>
                              ) : null}
                              {Array.isArray(summary.cooldown_opportunities) && summary.cooldown_opportunities.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Cooldown opportunities</div>
                                  <div className="mt-1 space-y-1">
                                    {summary.cooldown_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                      return renderAutonomySummaryRow(
                                        'domain',
                                        String(profile.id),
                                        'suppressed',
                                        row,
                                        idx,
                                        <>
                                          <div>{String(row.title || row.canonical_key || 'Cooldown opportunity')}{row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                          {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('domain', String(profile.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'domain', ownerId: String(profile.id) })}
                                        </>
                                      );
                                    })}
                                  </div>
                                </div>
                              ) : null}
                              {Array.isArray(summary.skipped_opportunities) && summary.skipped_opportunities.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Skipped opportunities</div>
                                  <div className="mt-1 space-y-1">
                                    {summary.skipped_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                      return renderAutonomySummaryRow(
                                        'domain',
                                        String(profile.id),
                                        'suppressed',
                                        row,
                                        idx,
                                        <>
                                          <div>{String(row.title || row.canonical_key || 'Skipped opportunity')}{row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                          {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('domain', String(profile.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'domain', ownerId: String(profile.id) })}
                                        </>
                                      );
                                    })}
                                  </div>
                                </div>
                              ) : null}
                              {summary.evidence_mix ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Evidence mix</div>
                                  <div className="mt-1 text-gray-600">
                                    KB {Number((summary.evidence_mix as any)?.documents || 0)}
                                    {' '}· Repo {Number((summary.evidence_mix as any)?.repo_documents || 0)}
                                    {' '}· Papers {Number((summary.evidence_mix as any)?.papers || 0)}
                                  </div>
                                </div>
                              ) : null}
                              {ideaTitles.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Latest top ideas</div>
                                  <div className="mt-1 space-y-1">
                                    {ideaTitles.map((idea) => (
                                      <div key={String(idea)}>{String(idea)}</div>
                                    ))}
                                  </div>
                                </div>
                              ) : null}
                              {opportunities.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Opportunity queue</div>
                                  <div className="mt-2 space-y-2">
                                    {opportunities.slice(0, 6).map((row) => {
                                      const opportunityRowKey = buildAutonomyOpportunityRowKey('domain', String(profile.id), String(row.opportunity_id));
                                      const opportunityNoteId = String((Array.isArray(row.source_note_ids) && row.source_note_ids.length > 0
                                        ? row.source_note_ids[0]
                                        : (Array.isArray(profile.latest_note_ids) && profile.latest_note_ids.length > 0 ? profile.latest_note_ids[0] : '')) || '').trim();
                                      return (
                                      <div
                                        key={row.opportunity_id}
                                        ref={registerAutonomyRowRef(opportunityRowKey)}
                                        className={`border border-gray-100 rounded p-2 transition-colors ${highlightedAutonomyRowKey === opportunityRowKey ? AUTONOMY_FOCUS_ROW_CLASS : ''}`}
                                      >
                                        <div className="flex items-center justify-between gap-2">
                                          <div className="font-medium text-gray-900">{row.title}</div>
                                          <span className={`text-[11px] px-2 py-0.5 rounded ${researchOpportunityStageClass(row.stage)}`}>
                                            {row.stage}
                                          </span>
                                        </div>
                                        <div className="mt-1 text-gray-500">
                                          Confidence {Number(row.confidence || 0).toFixed(2)}
                                          {' '}· Novelty {Number(row.novelty || 0).toFixed(2)}
                                          {' '}· Readiness {Number(row.readiness || 0).toFixed(2)}
                                        </div>
                                        {row.operator_note ? <div className="mt-1 text-gray-500">Note: {row.operator_note}</div> : null}
                                        <div className="mt-2 text-gray-500">
                                          Plans {Array.isArray(row.linked_experiment_plan_ids) ? row.linked_experiment_plan_ids.length : 0}
                                          {' '}· Runs {Array.isArray(row.linked_validation_run_ids) ? row.linked_validation_run_ids.length : 0}
                                          {' '}· Jobs {Array.isArray(row.child_job_ids) ? row.child_job_ids.length : 0}
                                        </div>
                                        {String(row.latest_experiment_plan_id || row.latest_validation_run_id || row.latest_validation_job_id || '').trim() ? (
                                          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-gray-500">
                                            {row.latest_experiment_plan_id ? <span>Latest plan {String(row.latest_experiment_plan_id).slice(0, 8)}</span> : null}
                                            {row.latest_validation_run_id ? <span>Run {String(row.latest_validation_run_id).slice(0, 8)}</span> : null}
                                            {row.latest_validation_status ? <span>Status {String(row.latest_validation_status).replace(/_/g, ' ')}</span> : null}
                                            {row.latest_validation_blocked_reason_code ? <span>Blocked {String(row.latest_validation_blocked_reason_code).replace(/_/g, ' ')}</span> : null}
                                            {row.latest_experiment_plan_id && opportunityNoteId ? (
                                              <Button
                                                size="sm"
                                                variant="ghost"
                                                className="!px-2 !py-1 !h-auto text-xs"
                                                onClick={() => navigate(buildResearchNoteExperimentUrl(opportunityNoteId, { plan: String(row.latest_experiment_plan_id) }))}
                                              >
                                                Open plan
                                              </Button>
                                            ) : null}
                                            {row.latest_validation_run_id && opportunityNoteId ? (
                                              <Button
                                                size="sm"
                                                variant="ghost"
                                                className="!px-2 !py-1 !h-auto text-xs"
                                                onClick={() => navigate(buildResearchNoteExperimentUrl(opportunityNoteId, { run: String(row.latest_validation_run_id) }))}
                                              >
                                                Open run
                                              </Button>
                                            ) : null}
                                            {row.latest_validation_job_id ? (
                                              <Button
                                                size="sm"
                                                variant="ghost"
                                                className="!px-2 !py-1 !h-auto text-xs"
                                                onClick={() => navigate(buildAutonomousAgentsUrl(String(row.latest_validation_job_id)), { replace: true })}
                                              >
                                                Open validation job
                                              </Button>
                                            ) : null}
                                          </div>
                                        ) : null}
                                        <div className="mt-1 text-gray-500">
                                          Autonomy {String(row.autonomy_state || 'eligible').replace(/_/g, ' ')}
                                          {row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : ''}
                                          {row.next_eligible_at ? ` · Next eligible ${new Date(row.next_eligible_at).toLocaleString()}` : ''}
                                        </div>
                                        {renderOpportunityReevaluationReviewMeta(row, (url) => navigate(url))}
                                        {renderOpportunityFollowUpOutcomeMeta(row)}
                                        {renderOpportunityExplainabilityPanel(opportunityRowKey, row, { surface: 'domain', ownerId: String(profile.id) })}
                                        <div className="mt-2 flex flex-wrap gap-2">
                                          {row.decision_state !== 'accepted' ? (
                                            <Button
                                              size="sm"
                                              variant="secondary"
                                              onClick={() => domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'accept' })}
                                            >
                                              Accept
                                            </Button>
                                          ) : null}
                                          {row.decision_state !== 'suppressed' ? (
                                            <Button
                                              size="sm"
                                              variant="ghost"
                                              onClick={() => beginOpportunitySuppression('domain', profile.id, row)}
                                            >
                                              Suppress
                                            </Button>
                                          ) : (
                                            <Button
                                              size="sm"
                                              variant="ghost"
                                              onClick={() => domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'reopen' })}
                                            >
                                              Reopen
                                            </Button>
                                          )}
                                          {row.decision_state === 'accepted' ? (
                                            <Button
                                              size="sm"
                                              variant="primary"
                                              onClick={() => domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'materialize_experiment', startImmediately: true })}
                                            >
                                              Run Experiment
                                            </Button>
                                          ) : null}
                                          <Button
                                            size="sm"
                                            variant="ghost"
                                            disabled={Array.isArray(row.linked_experiment_plan_ids) && row.linked_experiment_plan_ids.length > 0}
                                            onClick={() => domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'create_plan' })}
                                          >
                                            Create Plan
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="ghost"
                                            disabled={Array.isArray(row.linked_validation_run_ids) && row.linked_validation_run_ids.length > 0}
                                            onClick={() => domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'launch_validation' })}
                                          >
                                            Launch Validation
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="ghost"
                                            disabled={canRelaunchOpportunityRow(row) ? false : Array.isArray(row.child_job_ids) && row.child_job_ids.length > 0}
                                            onClick={() => (
                                              canRelaunchOpportunityRow(row)
                                                ? beginOpportunityRelaunch('domain', String(profile.id), row)
                                                : domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'launch_follow_up' })
                                            )}
                                          >
                                            {canRelaunchOpportunityRow(row) ? 'Relaunch Follow-up' : 'Follow-up'}
                                          </Button>
                                        </div>
                                        {opportunityNoteDraft?.surface === 'domain'
                                        && String(opportunityNoteDraft.ownerId) === String(profile.id)
                                        && String(opportunityNoteDraft.opportunityId) === String(row.opportunity_id) ? (
                                          <div className={`mt-2 rounded p-2 ${opportunityNoteDraft.mode === 'suppress' ? 'border border-rose-200 bg-rose-50' : 'border border-emerald-200 bg-emerald-50'}`}>
                                            <div className={`text-[11px] font-medium ${opportunityNoteDraft.mode === 'suppress' ? 'text-rose-700' : 'text-emerald-700'}`}>
                                              {opportunityNoteDraft.mode === 'suppress' ? 'Suppression note' : 'Relaunch note'}
                                            </div>
                                            <textarea
                                              aria-label={opportunityNoteDraft.mode === 'suppress' ? 'Domain suppression note' : 'Domain relaunch note'}
                                              className={`mt-2 w-full rounded px-2 py-1 text-xs ${opportunityNoteDraft.mode === 'suppress' ? 'border border-rose-200' : 'border border-emerald-200'}`}
                                              rows={3}
                                              value={opportunityNoteDraft.value}
                                              onChange={(e) => setOpportunityNoteDraft((prev) => prev ? { ...prev, value: e.target.value } : prev)}
                                            />
                                            <div className="mt-2 flex gap-2">
                                              <Button size="sm" variant="secondary" onClick={submitOpportunityAction}>
                                                {opportunityNoteDraft.mode === 'suppress' ? 'Save suppression' : 'Relaunch follow-up'}
                                              </Button>
                                              <Button size="sm" variant="ghost" onClick={cancelOpportunityAction}>
                                                Cancel
                                              </Button>
                                            </div>
                                          </div>
                                        ) : null}
                                      </div>
                                    );})}
                                  </div>
                                </div>
                              ) : null}
                              <SharedAutonomyReviewLists
                                sections={[
                                  { title: 'Queued operator reviews', rows: summary.queued_operator_reviews as Array<Record<string, any>> | undefined },
                                  {
                                    title: 'Pending approvals',
                                    rows: summary.pending_follow_up_approvals as Array<Record<string, any>> | undefined,
                                    renderRow: (row, idx) => renderInlineFollowUpApprovalRow('domain', String(profile.id), row, idx),
                                  },
                                  {
                                    title: 'Manual recommendations',
                                    rows: summary.manual_follow_up_recommendations as Array<Record<string, any>> | undefined,
                                    renderRow: (row, idx) => renderInlineManualRecommendationRow(
                                      'domain',
                                      String(profile.id),
                                      row,
                                      idx,
                                      opportunities as Array<Record<string, any>> | undefined,
                                    ),
                                  },
                                  {
                                    title: 'Suppressed relaunches',
                                    rows: summary.suppressed_relaunches as Array<Record<string, any>> | undefined,
                                    renderRow: (row, idx) => renderInlineSuppressedRelaunchRow(
                                      'domain',
                                      String(profile.id),
                                      row,
                                      idx,
                                      opportunities as Array<Record<string, any>> | undefined,
                                    ),
                                  },
                                ]}
                              />
                              {(Array.isArray(profile.latest_note_ids) && profile.latest_note_ids.length > 0) || (Array.isArray(profile.latest_experiment_plan_ids) && profile.latest_experiment_plan_ids.length > 0) || validationRuns.length > 0 || (Array.isArray(profile.latest_validation_run_ids) && profile.latest_validation_run_ids.length > 0) ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Artifacts</div>
                                  {Array.isArray(profile.latest_note_ids) && profile.latest_note_ids.length > 0 ? (
                                    <div className="mt-1 text-gray-600">Research notes: {profile.latest_note_ids.join(', ')}</div>
                                  ) : null}
                                  {Array.isArray(profile.latest_experiment_plan_ids) && profile.latest_experiment_plan_ids.length > 0 ? (
                                    <div className="mt-1 text-gray-600">Experiment plans: {profile.latest_experiment_plan_ids.join(', ')}</div>
                                  ) : null}
                                  {validationRuns.length > 0 ? (
                                    <div className="mt-2">{renderScientificValidationRuns(validationRuns as any, { ownerProfile: profile })}</div>
                                  ) : Array.isArray(profile.latest_validation_run_ids) && profile.latest_validation_run_ids.length > 0 ? (
                                    <div className="mt-1 text-gray-600">Validation runs: {profile.latest_validation_run_ids.join(', ')}</div>
                                  ) : null}
                                </div>
                              ) : null}
                            </div>
                          </details>
                        </div>
                      );
                    })}
                    {!(((domainProfilesData as any)?.items || []) as DomainResearchProfile[]).length ? (
                      <div className="text-sm text-gray-500">No domain profiles yet.</div>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'swarm' && (
          <div className="w-full flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Swarm Review</h2>
                <p className="text-sm text-gray-500">
                  Review unresolved coding swarms, compare candidate paths, and route the strongest path into repair or backlog.
                </p>
              </div>
              <div className="flex gap-2">
                <Button variant="ghost" size="sm" onClick={() => refetchSwarmReviewJobs()}>
                  <RefreshCw className="w-4 h-4 mr-1" />
                  Refresh jobs
                </Button>
                <Button variant="ghost" size="sm" onClick={() => refetchSwarmAnalytics()}>
                  <RefreshCw className="w-4 h-4 mr-1" />
                  Refresh analytics
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Total runs</div>
                <div className="mt-1 text-2xl font-semibold text-gray-900">{Number((swarmAnalyticsData as any)?.totals?.total_runs || 0)}</div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Repair handoffs</div>
                <div className="mt-1 text-2xl font-semibold text-emerald-700">{Number((swarmAnalyticsData as any)?.totals?.repair_handoff_runs || 0)}</div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Needs review</div>
                <div className="mt-1 text-2xl font-semibold text-amber-700">{Number((swarmAnalyticsData as any)?.totals?.review_needed_runs || 0)}</div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Avg confidence</div>
                <div className="mt-1 text-2xl font-semibold text-cyan-700">
                  {typeof (swarmAnalyticsData as any)?.totals?.avg_confidence === 'number'
                    ? `${(Number((swarmAnalyticsData as any).totals.avg_confidence) * 100).toFixed(0)}%`
                    : 'n/a'}
                </div>
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-3 mb-4">
              <div className="flex flex-wrap gap-3 items-center">
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmReviewVisibilityScope}
                  onChange={(e) => setSwarmReviewVisibilityScope(e.target.value as 'mine' | 'shared' | 'all')}
                >
                  <option value="mine">My items</option>
                  <option value="shared">Shared with me</option>
                  <option value="all">All visible</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmReviewPresetFilter}
                  onChange={(e) => setSwarmReviewPresetFilter(e.target.value)}
                >
                  <option value="">All presets</option>
                  <option value="bug_triage_swarm">Bug Triage</option>
                  <option value="build_break_swarm">Build Break</option>
                  <option value="frontend_regression_swarm">Frontend Regression</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmReviewStateFilter}
                  onChange={(e) => setSwarmReviewStateFilter(e.target.value)}
                >
                  <option value="">All review states</option>
                  <option value="needs_review">Needs review</option>
                  <option value="insufficient_swarm_consensus">Insufficient consensus</option>
                  <option value="consensus_failed">Consensus failed</option>
                  <option value="tie_break_running">Tie-break running</option>
                  <option value="manual_promotion">Manual promotion</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmReviewConfidenceBand}
                  onChange={(e) => setSwarmReviewConfidenceBand(e.target.value)}
                >
                  <option value="">Any confidence band</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmReviewBacklogFilter}
                  onChange={(e) => setSwarmReviewBacklogFilter(e.target.value)}
                >
                  <option value="">Any backlog status</option>
                  <option value="linked">Already sent to backlog</option>
                  <option value="unlinked">Not yet in backlog</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmReviewAssignmentFilter}
                  onChange={(e) => setSwarmReviewAssignmentFilter(e.target.value)}
                >
                  <option value="">Any assignment</option>
                  <option value="assigned_to_me">Assigned to me</option>
                  <option value="unassigned">Unassigned</option>
                  {collaborationUsers.map((candidate) => (
                    <option key={String(candidate.id)} value={String(candidate.id)}>
                      {userLabelById(String(candidate.id))}
                    </option>
                  ))}
                </select>
                {(swarmReviewPresetFilter || swarmReviewStateFilter || swarmReviewConfidenceBand || swarmReviewBacklogFilter || swarmReviewAssignmentFilter || swarmReviewVisibilityScope !== 'mine') ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setSwarmReviewVisibilityScope('mine');
                      setSwarmReviewPresetFilter('');
                      setSwarmReviewStateFilter('');
                      setSwarmReviewConfidenceBand('');
                      setSwarmReviewBacklogFilter('');
                      setSwarmReviewAssignmentFilter('');
                    }}
                  >
                    <XCircle className="w-4 h-4 mr-1" />
                    Clear
                  </Button>
                ) : null}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 mb-4">
              {(((swarmAnalyticsData as any)?.preset_rows || []) as Array<Record<string, any>>).map((row) => (
                <div key={String(row.preset_key || row.launch_mode)} className="bg-white border border-gray-200 rounded-lg p-3">
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-medium text-gray-900">{String(row.label || row.preset_key)}</div>
                    <span className="text-xs bg-slate-100 text-slate-700 px-2 py-1 rounded">
                      {Number(row.total_runs || 0)} runs
                    </span>
                  </div>
                  <div className="mt-2 text-sm text-gray-600">
                    Confidence {typeof row.avg_confidence === 'number' ? `${(Number(row.avg_confidence) * 100).toFixed(0)}%` : 'n/a'}
                  </div>
                  <div className="mt-2 text-xs text-gray-500">
                    Promotion {(Number(row.promotion_rate || 0) * 100).toFixed(0)}% · Review {(Number(row.review_rate || 0) * 100).toFixed(0)}% · Tie-break {(Number(row.tie_breaker_rate || 0) * 100).toFixed(0)}%
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2 text-xs">
                    <span className="bg-emerald-50 text-emerald-700 px-2 py-1 rounded">Repair {Number(row.repair_handoff_runs || 0)}</span>
                    <span className="bg-amber-50 text-amber-700 px-2 py-1 rounded">Review {Number(row.review_needed_runs || 0)}</span>
                    <span className="bg-slate-100 text-slate-700 px-2 py-1 rounded">Backlog {Number(row.backlog_handoff_runs || 0)}</span>
                    <span className="bg-amber-100 text-amber-800 px-2 py-1 rounded">Auto backlog {Number(row.auto_backlog_handoff_runs || 0)}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-1">
              {swarmReviewJobsLoading || swarmAnalyticsLoading ? (
                <div className="flex justify-center items-center py-12">
                  <LoadingSpinner />
                </div>
              ) : filteredSwarmReviewJobs.length === 0 ? (
                <div className="text-sm text-gray-500">No swarm review jobs match the current filters.</div>
              ) : (
                filteredSwarmReviewJobs.map((job) => {
                  const cfg = (job.config || {}) as Record<string, any>;
                  const quickStart = (cfg.quick_start && typeof cfg.quick_start === 'object') ? (cfg.quick_start as Record<string, any>) : {};
                  const swarmSummary = (((job as any)?.swarm_summary && typeof (job as any).swarm_summary === 'object')
                    ? ((job as any).swarm_summary as Record<string, any>)
                    : {}) as Record<string, any>;
                  const presetKey = String(quickStart.preset_key || cfg.coding_swarm_preset_key || '').trim().toLowerCase();
                  const presetLabel = presetKey === 'build_break_swarm'
                    ? 'Build Break Swarm'
                    : presetKey === 'frontend_regression_swarm'
                      ? 'Frontend Regression Swarm'
                      : 'Bug Triage Swarm';
                  const reviewState = String(swarmSummary.review_state || '').trim() || 'needs_review';
                  const reviewReason = String(swarmSummary.review_reason || swarmSummary.promotion_reason || '').trim();
                  const confidenceOverall = Number((swarmSummary.confidence as any)?.overall || 0);
                  const candidatePaths = Array.isArray(swarmSummary.candidate_paths) ? swarmSummary.candidate_paths : [];
                  const linkedBacklogItems = backlogBySwarmJobId[String(job.id)] || [];
                  const linkedBacklogRouteMode = String((((linkedBacklogItems[0] as any)?.lineage || {}) as Record<string, any>).originating_swarm_route_mode || '').trim().toLowerCase();
                  const swarmCollaborationSummary = ((swarmSummary.collaboration_summary && typeof swarmSummary.collaboration_summary === 'object')
                    ? swarmSummary.collaboration_summary
                    : {}) as Record<string, any>;
                  const reviewNote = String(swarmSummary.review_note || '').trim();
                  const swarmReviewNoteValue = swarmReviewNoteDrafts[String(job.id)] ?? reviewNote;
                  return (
                    <div key={String(job.id)} className="bg-white border border-gray-200 rounded-lg p-4">
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <div className="font-medium text-gray-900">{job.name}</div>
                            <span className="text-xs px-2 py-1 rounded bg-rose-50 text-rose-700 border border-rose-100">{presetLabel}</span>
                            <span className="text-xs px-2 py-1 rounded bg-slate-100 text-slate-700 border border-slate-200">{reviewState.replace(/_/g, ' ')}</span>
                            {typeof confidenceOverall === 'number' ? (
                              <span className="text-xs px-2 py-1 rounded bg-cyan-50 text-cyan-700 border border-cyan-100">
                                Confidence {(confidenceOverall * 100).toFixed(0)}%
                              </span>
                            ) : null}
                            {linkedBacklogItems.length > 0 ? (
                              <span className="text-xs px-2 py-1 rounded bg-amber-50 text-amber-700 border border-amber-100">
                                {linkedBacklogRouteMode === 'auto' ? 'Auto-routed to backlog' : 'Backlog linked'} {linkedBacklogItems.length}
                              </span>
                            ) : null}
                          </div>
                          <div className="mt-1 text-sm text-gray-600">{String(job.goal || '').slice(0, 220)}</div>
                          {reviewReason ? (
                            <div className="mt-2 text-xs text-gray-500">{reviewReason}</div>
                          ) : null}
                          <CollaborationSummaryPanel
                            summary={swarmCollaborationSummary as CollaborationSummary}
                            fallbackOwnerId={String(swarmSummary.owner_user_id || job.user_id || '')}
                            fallbackVisibility={String(swarmCollaborationSummary.visibility_scope || (Array.isArray(swarmSummary.shared_with_user_ids) && swarmSummary.shared_with_user_ids.length > 0 ? 'shared' : 'private'))}
                            fallbackSharedWithUserIds={Array.isArray(swarmSummary.shared_with_user_ids) ? swarmSummary.shared_with_user_ids.map((value: unknown) => String(value || '').trim()).filter(Boolean) : []}
                            userLabelById={userLabelById}
                            assigneeUsers={collaborationUsers}
                            showAssigneeSelect
                            assigneeValue={String(swarmSummary.assigned_user_id || '')}
                            onAssigneeChange={(nextAssignee) => {
                              if (!nextAssignee) {
                                actionMutation.mutate({ jobId: job.id, action: 'clear_swarm_assignment' });
                              } else {
                                actionMutation.mutate({ jobId: job.id, action: 'assign_swarm_review', actionPayload: { assigned_user_id: nextAssignee } });
                              }
                            }}
                            onClearAssignee={() => actionMutation.mutate({ jobId: job.id, action: 'clear_swarm_assignment' })}
                            noteValue={swarmReviewNoteValue}
                            onNoteChange={(value) =>
                              setSwarmReviewNoteDrafts((prev) => ({
                                ...prev,
                                [String(job.id)]: value,
                              }))
                            }
                            onNoteSave={() =>
                              actionMutation.mutate({
                                jobId: job.id,
                                action: 'update_swarm_review_note',
                                actionPayload: { review_note: swarmReviewNoteValue },
                              })
                            }
                            noteSaveLabel="Save review note"
                            notePlaceholder="Swarm review note"
                          />
                          <div className="mt-2 text-xs text-gray-500 flex flex-wrap gap-3">
                            <span>Repo {String(quickStart.source_name || cfg.source_id || 'unknown')}</span>
                            {swarmSummary.winning_role ? <span>Winning role {String(swarmSummary.winning_role)}</span> : null}
                            {swarmSummary.repair_chain_job_id ? <span>Repair handoff {String(swarmSummary.repair_chain_job_id).slice(0, 8)}</span> : null}
                          </div>
                        </div>
                        <div className="flex flex-wrap gap-2 shrink-0">
                          <Button size="sm" variant="ghost" onClick={() => { setSelectedJob(job); setActiveTab('jobs'); }}>
                            Open job
                          </Button>
                          <Button
                            size="sm"
                            variant="secondary"
                            disabled={actionMutation.isLoading || !!swarmSummary.repair_chain_job_id}
                            onClick={() => actionMutation.mutate({ jobId: job.id, action: 'launch_tie_breaker' })}
                          >
                            Relaunch verifier
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={actionMutation.isLoading}
                            onClick={() => actionMutation.mutate({ jobId: job.id, action: 'assign_swarm_review', actionPayload: { assigned_user_id: String(user?.id || '') } })}
                          >
                            Assign to me
                          </Button>
                          <Button
                            size="sm"
                            variant="primary"
                            disabled={actionMutation.isLoading || !candidatePaths.length || !!swarmSummary.repair_chain_job_id}
                            onClick={() =>
                              actionMutation.mutate({
                                jobId: job.id,
                                action: 'promote_swarm_candidate',
                                actionPayload: {
                                  candidate_job_id: String((candidatePaths[0] as any)?.job_id || ''),
                                },
                              })
                            }
                          >
                            Promote top path
                          </Button>
                        </div>
                      </div>
                      {candidatePaths.length > 0 ? (
                        <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-3">
                          {candidatePaths.slice(0, 4).map((candidate: any, idx: number) => (
                            <div key={`${String(candidate.job_id || 'candidate')}-${idx}`} className="border border-gray-200 rounded-lg p-3 bg-slate-50">
                              <div className="flex items-center justify-between gap-2">
                                <div className="font-medium text-gray-900">{String(candidate.role || 'Candidate')}</div>
                                <div className="text-xs text-gray-500">Score {Number(candidate.score || 0).toFixed(2)}</div>
                              </div>
                              {Array.isArray(candidate.suspect_files) && candidate.suspect_files.length > 0 ? (
                                <div className="mt-2 text-xs text-gray-600">
                                  Files: {candidate.suspect_files.slice(0, 4).map((value: any) => String(value || '')).join(', ')}
                                </div>
                              ) : null}
                              {Array.isArray(candidate.recommended_commands) && candidate.recommended_commands.length > 0 ? (
                                <div className="mt-2 text-xs text-gray-600">
                                  Commands: {candidate.recommended_commands.slice(0, 2).map((value: any) => String(value || '')).join(' | ')}
                                </div>
                              ) : null}
                              <div className="mt-3 flex gap-2">
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  disabled={actionMutation.isLoading || !!swarmSummary.repair_chain_job_id}
                                  onClick={() =>
                                    actionMutation.mutate({
                                      jobId: job.id,
                                      action: 'promote_swarm_candidate',
                                      actionPayload: {
                                        candidate_job_id: String(candidate.job_id || ''),
                                        candidate_index: idx,
                                      },
                                    })
                                  }
                                >
                                  Promote this path
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : null}

                      {linkedBacklogItems.length > 0 ? (
                        <div className="mt-3 text-xs text-gray-600">
                          {linkedBacklogRouteMode === 'auto' ? 'Auto-routed backlog' : 'Backlog'}: {linkedBacklogItems.map((item) => String(item.title || item.id)).slice(0, 2).join(' · ')}
                        </div>
                      ) : (
                        <div className="mt-3">
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={createCodingBacklogMutation.isLoading || !String(cfg.source_id || '').trim()}
                            onClick={() => {
                              const topCandidate = (candidatePaths[0] || {}) as Record<string, any>;
                              createCodingBacklogMutation.mutate({
                                title: `${presetLabel} review - ${String(job.name || 'autonomous job').slice(0, 72)}`,
                                portfolio_goal: String(job.goal || 'Review coding swarm findings and implement the best repair path').slice(0, 2000),
                                source_id: String(cfg.source_id || ''),
                                scope: String(cfg.scope || 'auto') || 'auto',
                                failure_symptom: String(cfg.failure_symptom || '').trim() || undefined,
                                error_output: String(cfg.error_output || '').trim() || undefined,
                                file_paths: Array.from(new Set((Array.isArray(topCandidate.suspect_files) ? topCandidate.suspect_files : []).map((value) => String(value || '').trim()).filter(Boolean))).slice(0, 12),
                                commands: Array.isArray(topCandidate.recommended_commands) ? topCandidate.recommended_commands.slice(0, 6).map((value: any) => String(value || '').trim()).filter(Boolean) : [],
                                visibility: Array.isArray(swarmSummary?.shared_with_user_ids) && swarmSummary.shared_with_user_ids.length > 0 ? 'shared' : 'private',
                                shared_with_user_ids: Array.isArray(swarmSummary?.shared_with_user_ids) ? swarmSummary.shared_with_user_ids.slice(0, 200).map((value) => String(value || '').trim()).filter(Boolean) : [],
                                assigned_user_id: String(swarmSummary?.assigned_user_id || '').trim() || undefined,
                                assigned_by_user_id: String(swarmSummary?.assigned_by_user_id || '').trim() || undefined,
                                assigned_at: String(swarmSummary?.assigned_at || '').trim() || undefined,
                                collaboration: {
                                  owner_user_id: String(swarmSummary?.owner_user_id || job.user_id || '').trim() || undefined,
                                  visibility: Array.isArray(swarmSummary?.shared_with_user_ids) && swarmSummary.shared_with_user_ids.length > 0 ? 'shared' : 'private',
                                  shared_with_user_ids: Array.isArray(swarmSummary?.shared_with_user_ids) ? swarmSummary.shared_with_user_ids.slice(0, 200).map((value) => String(value || '').trim()).filter(Boolean) : [],
                                  assigned_user_id: String(swarmSummary?.assigned_user_id || '').trim() || undefined,
                                  assigned_by_user_id: String(swarmSummary?.assigned_by_user_id || '').trim() || undefined,
                                  assigned_at: String(swarmSummary?.assigned_at || '').trim() || undefined,
                                  note: reviewReason || undefined,
                                },
                                lineage: {
                                  originating_swarm_job_id: String(job.id || ''),
                                  originating_swarm_preset: presetKey || undefined,
                                  originating_swarm_review_reason: reviewReason || undefined,
                                  originating_swarm_candidate_job_id: String(topCandidate.job_id || '').trim() || undefined,
                                  originating_swarm_candidate_role: String(topCandidate.role || '').trim() || undefined,
                                  originating_swarm_candidate_index: 0,
                                  originating_swarm_route_mode: 'manual',
                                },
                                start_immediately: false,
                              });
                            }}
                          >
                            <Layers className="w-4 h-4 mr-1" />
                            Send to backlog
                          </Button>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {activeTab === 'outcomes' && (
          <div className="w-full flex flex-col min-h-0 gap-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Swarm Outcomes</h2>
                <p className="text-sm text-gray-500">
                  Track the coding swarm funnel from promotion through repair, verification, and backlog routing.
                </p>
              </div>
              <Button variant="ghost" size="sm" onClick={() => refetchSwarmOutcomeAnalytics()}>
                <RefreshCw className="w-4 h-4 mr-1" />
                Refresh outcomes
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Swarm roots</div>
                <div className="mt-1 text-2xl font-semibold text-gray-900">
                  {Number((swarmOutcomeAnalyticsData as any)?.totals?.total_swarm_roots || 0)}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Repair handoffs</div>
                <div className="mt-1 text-2xl font-semibold text-cyan-700">
                  {Number((swarmOutcomeAnalyticsData as any)?.totals?.repair_handoff_runs || 0)}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Verified fixes</div>
                <div className="mt-1 text-2xl font-semibold text-emerald-700">
                  {Number((swarmOutcomeAnalyticsData as any)?.totals?.verified_fix_runs || 0)}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Backlog routes</div>
                <div className="mt-1 text-2xl font-semibold text-amber-700">
                  {Number((swarmOutcomeAnalyticsData as any)?.totals?.backlog_routed_runs || 0)}
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wide text-gray-500">Avg handoff</div>
                <div className="mt-1 text-2xl font-semibold text-violet-700">
                  {typeof (swarmOutcomeAnalyticsData as any)?.totals?.avg_handoff_minutes === 'number'
                    ? `${Number((swarmOutcomeAnalyticsData as any).totals.avg_handoff_minutes).toFixed(0)}m`
                    : 'n/a'}
                </div>
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-3">
              <div className="flex flex-wrap gap-3 items-center">
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmOutcomeVisibilityScope}
                  onChange={(e) => setSwarmOutcomeVisibilityScope(e.target.value as 'mine' | 'shared' | 'all')}
                >
                  <option value="mine">My items</option>
                  <option value="shared">Shared with me</option>
                  <option value="all">All visible</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmOutcomePresetFilter}
                  onChange={(e) => setSwarmOutcomePresetFilter(e.target.value)}
                >
                  <option value="">All presets</option>
                  <option value="bug_triage_swarm">Bug Triage</option>
                  <option value="build_break_swarm">Build Break</option>
                  <option value="frontend_regression_swarm">Frontend Regression</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmOutcomeTerminalFilter}
                  onChange={(e) => setSwarmOutcomeTerminalFilter(e.target.value)}
                >
                  <option value="">All outcomes</option>
                  <option value="verified_fix">Verified fix</option>
                  <option value="repair_failed">Repair failed</option>
                  <option value="backlog_routed">Backlog routed</option>
                  <option value="needs_review">Needs review</option>
                  <option value="stalled_after_handoff">Stalled after handoff</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmOutcomePromotionFilter}
                  onChange={(e) => setSwarmOutcomePromotionFilter(e.target.value)}
                >
                  <option value="">Any promotion mode</option>
                  <option value="auto">Auto promotion</option>
                  <option value="manual">Manual promotion</option>
                  <option value="none">No promotion</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmOutcomeDateRange}
                  onChange={(e) => setSwarmOutcomeDateRange(e.target.value)}
                >
                  <option value="all">All time</option>
                  <option value="30d">Last 30 days</option>
                  <option value="7d">Last 7 days</option>
                </select>
                {(swarmOutcomePresetFilter || swarmOutcomeTerminalFilter || swarmOutcomePromotionFilter || swarmOutcomeDateRange !== 'all' || swarmOutcomeVisibilityScope !== 'mine') ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setSwarmOutcomeVisibilityScope('mine');
                      setSwarmOutcomePresetFilter('');
                      setSwarmOutcomeTerminalFilter('');
                      setSwarmOutcomePromotionFilter('');
                      setSwarmOutcomeDateRange('all');
                    }}
                  >
                    <XCircle className="w-4 h-4 mr-1" />
                    Clear
                  </Button>
                ) : null}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
              {(((swarmOutcomeAnalyticsData as any)?.preset_rows || []) as Array<Record<string, any>>).map((row) => (
                <div key={String(row.preset_key || row.launch_mode)} className="bg-white border border-gray-200 rounded-lg p-3">
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-medium text-gray-900">{String(row.label || row.preset_key)}</div>
                    <span className="text-xs bg-slate-100 text-slate-700 px-2 py-1 rounded">
                      {Number(row.total_swarm_roots || 0)} roots
                    </span>
                  </div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                    <span className="bg-emerald-50 text-emerald-700 px-2 py-1 rounded">Verified {Number(row.verified_fix_runs || 0)}</span>
                    <span className="bg-cyan-50 text-cyan-700 px-2 py-1 rounded">Repair {Number(row.repair_handoff_runs || 0)}</span>
                    <span className="bg-amber-50 text-amber-700 px-2 py-1 rounded">Backlog {Number(row.backlog_routed_runs || 0)}</span>
                    <span className="bg-rose-50 text-rose-700 px-2 py-1 rounded">Failed {Number(row.repair_failed_runs || 0)}</span>
                  </div>
                  <div className="mt-2 text-xs text-gray-500">
                    Auto {Number(row.auto_promoted_runs || 0)} · Manual {Number(row.manual_promoted_runs || 0)} · Review {Number(row.needs_review_runs || 0)}
                  </div>
                  <div className="mt-1 text-xs text-gray-500">
                    Avg confidence {typeof row.avg_confidence === 'number' ? `${(Number(row.avg_confidence) * 100).toFixed(0)}%` : 'n/a'} · Avg handoff {typeof row.avg_handoff_minutes === 'number' ? `${Number(row.avg_handoff_minutes).toFixed(0)}m` : 'n/a'}
                  </div>
                  <div className="mt-1 text-xs text-gray-500">
                    Auto backlog {Number(row.auto_backlog_routed_runs || 0)} · Manual backlog {Number(row.manual_backlog_routed_runs || 0)} · Suppressed {Number(row.backlog_auto_suppressed_runs || 0)}
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4 flex-1 min-h-0">
              <div className="flex items-center justify-between mb-3">
                <h3 className="section-heading">Recent Cases</h3>
                <div className="text-xs text-gray-500">
                  {swarmOutcomeCases.length} cases
                </div>
              </div>
              {swarmOutcomeAnalyticsLoading ? (
                <div className="flex justify-center items-center h-40"><LoadingSpinner /></div>
              ) : swarmOutcomeCases.length === 0 ? (
                <div className="text-sm text-gray-500">No coding swarm outcome cases match the current filters.</div>
              ) : (
                <div className="space-y-3 max-h-[42rem] overflow-y-auto pr-1">
                  {swarmOutcomeCases.map((item) => {
                    const collaborationSummary = ((item.collaboration_summary && typeof item.collaboration_summary === 'object')
                      ? item.collaboration_summary
                      : {}) as Record<string, any>;
                    return (
                    <div key={String(item.swarm_job_id)} className="border border-gray-200 rounded-lg p-3">
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <div className="font-medium text-gray-900">{String(item.swarm_job_name || item.swarm_job_id)}</div>
                            <span className="text-xs px-2 py-1 rounded bg-slate-100 text-slate-700 border border-slate-200">
                              {humanizeSwarmOutcome(item.preset_key)}
                            </span>
                            <span className={`text-xs px-2 py-1 rounded ${swarmOutcomeBadgeClass(item.terminal_outcome)}`}>
                              {humanizeSwarmOutcome(item.terminal_outcome)}
                            </span>
                            <span className="text-xs px-2 py-1 rounded bg-violet-50 text-violet-700 border border-violet-100">
                              Promotion {humanizeSwarmOutcome(item.promotion_mode)}
                            </span>
                          </div>
                          <div className="mt-2 text-xs text-gray-500 flex flex-wrap gap-3">
                            {item.source_label ? <span>Repo {String(item.source_label)}</span> : null}
                            {item.owner_user_id ? <span>Owner {String(collaborationSummary.owner_label || userLabelById(String(item.owner_user_id)) || String(item.owner_user_id).slice(0, 8))}</span> : null}
                            {item.assigned_user_id ? <span>Assignee {String(collaborationSummary.assignee_label || userLabelById(String(item.assigned_user_id)) || String(item.assigned_user_id).slice(0, 8))}</span> : null}
                            <span>Visibility {humanizeDecisionTraceValue(String(collaborationSummary.visibility_scope || 'private'))}</span>
                            {Number((collaborationSummary.shared_with_user_ids || []).length || 0) > 0 ? <span>Shared with {Number((collaborationSummary.shared_with_user_ids || []).length || 0)}</span> : null}
                            {item.repair_job_id ? <span>Repair {String(item.repair_status || 'linked')}</span> : null}
                            {item.verification_status ? <span>Verification {humanizeSwarmOutcome(item.verification_status)}</span> : null}
                            {item.backlog_item_id ? <span>Backlog {String(item.backlog_route_mode || 'linked')} · {String(item.backlog_status || 'linked')}</span> : null}
                            {typeof item.handoff_latency_minutes === 'number' ? <span>Handoff {Number(item.handoff_latency_minutes).toFixed(0)}m</span> : null}
                          </div>
                          {item.review_note ? (
                            <div className="mt-1 text-xs text-slate-600">Note: {String(item.review_note)}</div>
                          ) : null}
                          {item.terminal_reason ? (
                            <div className="mt-2 text-xs text-gray-600">{String(item.terminal_reason)}</div>
                          ) : null}
                          {item.review_reason ? (
                            <div className="mt-1 text-xs text-gray-500">{String(item.review_reason)}</div>
                          ) : null}
                        </div>
                        <div className="flex flex-wrap gap-2 shrink-0">
                          <Button size="sm" variant="ghost" onClick={() => { setSelectedJob(null); navigate(buildAutonomousAgentsUrl(item.swarm_job_id)); setActiveTab('jobs'); }}>
                            Open swarm
                          </Button>
                          {item.repair_job_id ? (
                            <Button size="sm" variant="ghost" onClick={() => { setSelectedJob(null); navigate(buildAutonomousAgentsUrl(String(item.repair_job_id))); setActiveTab('jobs'); }}>
                              Open repair
                            </Button>
                          ) : null}
                          {item.backlog_item_id ? (
                            <Button size="sm" variant="ghost" onClick={() => setActiveTab('backlog')}>
                              Open backlog
                            </Button>
                          ) : null}
                        </div>
                      </div>
                    </div>
                  )})}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'profiles' && (
          <div className="w-full flex flex-col min-h-0 gap-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Coding Swarm Profiles</h2>
                <p className="text-sm text-gray-500">
                  Save, edit, duplicate, and launch repo-scoped coding swarm presets for repeat triage work.
                </p>
              </div>
              <div className="flex gap-2">
                <Button variant="ghost" size="sm" onClick={() => queryClient.invalidateQueries(['coding-swarm-profiles'])}>
                  <RefreshCw className="w-4 h-4 mr-1" />
                  Refresh
                </Button>
                <Button
                  size="sm"
                  variant="primary"
                  onClick={() => openCodingSwarmProfileEditor(null)}
                >
                  New profile
                </Button>
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-3">
              <div className="flex flex-wrap gap-3 items-center">
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={profileOwnershipFilter}
                  onChange={(e) => setProfileOwnershipFilter(e.target.value)}
                >
                  <option value="">Mine + shared</option>
                  <option value="mine">Mine</option>
                  <option value="shared">Shared with me</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={profilePresetFilter}
                  onChange={(e) => setProfilePresetFilter(e.target.value)}
                >
                  <option value="">All presets</option>
                  <option value="bug_triage_swarm">Bug Triage</option>
                  <option value="build_break_swarm">Build Break</option>
                  <option value="frontend_regression_swarm">Frontend Regression</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={profileSourceFilter}
                  onChange={(e) => setProfileSourceFilter(e.target.value)}
                >
                  <option value="">All repos</option>
                  {codeSources.map((source: any) => (
                    <option key={String(source.id)} value={String(source.id)}>
                      {String(source.name || source.id)}
                    </option>
                  ))}
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={profileStatusFilter}
                  onChange={(e) => setProfileStatusFilter(e.target.value)}
                >
                  <option value="">Any status</option>
                  <option value="active">Active</option>
                  <option value="disabled">Disabled</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={profileVisibilityFilter}
                  onChange={(e) => setProfileVisibilityFilter(e.target.value)}
                >
                  <option value="">Any visibility</option>
                  <option value="private">Private</option>
                  <option value="shared">Shared</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={profileOwnerFilter}
                  onChange={(e) => setProfileOwnerFilter(e.target.value)}
                >
                  <option value="">Any owner</option>
                  {collaborationUsers.map((candidate) => (
                    <option key={String(candidate.id)} value={String(candidate.id)}>
                      {userLabelById(String(candidate.id))}
                    </option>
                  ))}
                </select>
                <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    className="rounded border-gray-300"
                    checked={profileDefaultOnly}
                    onChange={(e) => setProfileDefaultOnly(e.target.checked)}
                  />
                  Default only
                </label>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 min-h-0">
              <div className="xl:col-span-2 bg-white border border-gray-200 rounded-lg p-4 min-h-0">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="section-heading">Saved Profiles</h3>
                  <div className="text-xs text-gray-500">{filteredCodingSwarmProfiles.length} profiles</div>
                </div>
                <div className="space-y-3 max-h-[42rem] overflow-y-auto pr-1">
                  {filteredCodingSwarmProfiles.length === 0 ? (
                    <div className="text-sm text-gray-500">No coding swarm profiles match the current filters.</div>
                  ) : (
                    filteredCodingSwarmProfiles.map((profile) => {
                      const sourceLabel = String(codeSourceById[String(profile.source_id || '')]?.name || profile.source_id || '').trim();
                      const isOwner = String(profile.user_id || '') === String(user?.id || '');
                      const profileCollaborationSummary = ((profile.collaboration_summary && typeof profile.collaboration_summary === 'object')
                        ? profile.collaboration_summary
                        : {}) as CollaborationSummary;
                      return (
                        <div key={String(profile.id)} className="border border-gray-200 rounded-lg p-4">
                          <div className="flex items-start justify-between gap-4">
                            <div className="min-w-0">
                              <div className="flex flex-wrap items-center gap-2">
                                <div className="font-medium text-gray-900">{profile.title}</div>
                                <span className="text-xs px-2 py-1 rounded bg-rose-50 text-rose-700 border border-rose-100">
                                  {codingSwarmPresetLabel(profile.preset_key)}
                                </span>
                                <span className={`text-xs px-2 py-1 rounded ${String(profile.status || '').toLowerCase() === 'active' ? 'bg-emerald-50 text-emerald-700 border border-emerald-100' : 'bg-slate-100 text-slate-700 border border-slate-200'}`}>
                                  {String(profile.status || 'active')}
                                </span>
                                <span className={`text-xs px-2 py-1 rounded ${String(profile.visibility || 'private').toLowerCase() === 'shared' ? 'bg-cyan-50 text-cyan-700 border border-cyan-100' : 'bg-slate-100 text-slate-700 border border-slate-200'}`}>
                                  {String(profile.visibility || 'private')}
                                </span>
                                {profile.is_default ? (
                                  <span className="text-xs px-2 py-1 rounded bg-amber-50 text-amber-700 border border-amber-100">Default</span>
                                ) : null}
                              </div>
                              {profile.description ? (
                                <div className="mt-1 text-sm text-gray-600">{String(profile.description)}</div>
                              ) : null}
                              <CollaborationSummaryPanel
                                summary={profileCollaborationSummary}
                                fallbackOwnerId={String(profile.user_id || '')}
                                fallbackVisibility={String(profile.visibility || 'private')}
                                fallbackSharedWithUserIds={Array.isArray(profile.shared_with_user_ids) ? profile.shared_with_user_ids : []}
                                userLabelById={userLabelById}
                              />
                              <div className="mt-2 text-xs text-gray-500 flex flex-wrap gap-3">
                                <span>Repo {sourceLabel}</span>
                                <span>Scope {String(profile.scope_default || 'auto')}</span>
                                <span>Agents {Number(profile.max_agents || 4)}</span>
                                <span>Policy {String(profile.safe_command_policy || 'standard')}</span>
                                {profile.saved_search_query ? <span>Query saved</span> : null}
                                <span>Updated {new Date(profile.updated_at).toLocaleDateString()}</span>
                              </div>
                              {(profile.default_commands?.length || profile.default_file_paths?.length) ? (
                                <div className="mt-2 text-xs text-gray-500">
                                  {profile.default_commands?.length ? `Commands ${profile.default_commands.length}` : 'No commands'}
                                  {' · '}
                                  {profile.default_file_paths?.length ? `Files ${profile.default_file_paths.length}` : 'No files'}
                                </div>
                              ) : null}
                            </div>
                            <div className="flex flex-wrap gap-2 shrink-0">
                              <Button size="sm" variant="ghost" onClick={() => openCodingSwarmProfileEditor(profile)} disabled={!isOwner}>
                                Edit
                              </Button>
                              <Button size="sm" variant="ghost" onClick={() => openCodingSwarmProfileEditor(profile, { duplicate: true })}>
                                Duplicate
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() =>
                                  updateCodingSwarmProfileMutation.mutate({
                                    profileId: String(profile.id),
                                    data: { is_default: true, status: 'active' },
                                  })
                                }
                                disabled={updateCodingSwarmProfileMutation.isLoading || profile.is_default || !isOwner}
                              >
                                Set default
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() =>
                                  updateCodingSwarmProfileMutation.mutate({
                                    profileId: String(profile.id),
                                    data: { status: String(profile.status || '').toLowerCase() === 'active' ? 'disabled' : 'active' },
                                  })
                                }
                                disabled={updateCodingSwarmProfileMutation.isLoading || !isOwner}
                              >
                                {String(profile.status || '').toLowerCase() === 'active' ? 'Disable' : 'Enable'}
                              </Button>
                              <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => {
                                  setCodingSwarmLaunchSeed({
                                    presetKey: String(profile.preset_key || ''),
                                    profileId: String(profile.id),
                                    sourceId: String(profile.source_id || ''),
                                  });
                                  if (String(profile.preset_key || '') === 'build_break_swarm') setShowBuildBreakSwarmQuickStartModal(true);
                                  else if (String(profile.preset_key || '') === 'frontend_regression_swarm') setShowFrontendRegressionSwarmQuickStartModal(true);
                                  else setShowBugTriageSwarmQuickStartModal(true);
                                }}
                              >
                                Launch
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => deleteCodingSwarmProfileMutation.mutate(String(profile.id))}
                                disabled={deleteCodingSwarmProfileMutation.isLoading || !isOwner}
                              >
                                Delete
                              </Button>
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="section-heading">
                    {editingCodingSwarmProfileId ? 'Edit Profile' : codingSwarmProfileDraft.duplicate_mode ? 'Duplicate Profile' : 'New Profile'}
                  </h3>
                  {(editingCodingSwarmProfileId || codingSwarmProfileDraft.title) ? (
                    <Button size="sm" variant="ghost" onClick={closeCodingSwarmProfileEditor}>
                      Clear
                    </Button>
                  ) : null}
                </div>
                <div className="space-y-3">
                  <input
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    placeholder="Profile title"
                    value={codingSwarmProfileDraft.title}
                    onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, title: e.target.value }))}
                  />
                  <textarea
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    rows={3}
                    placeholder="Description"
                    value={String(codingSwarmProfileDraft.description || '')}
                    onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, description: e.target.value }))}
                  />
                  <select
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={String(codingSwarmProfileDraft.source_id || '')}
                    onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, source_id: e.target.value }))}
                    disabled={Boolean(editingCodingSwarmProfileId && !codingSwarmProfileDraft.duplicate_mode)}
                  >
                    <option value="">Select repo source</option>
                    {codeSources.map((source: any) => (
                      <option key={String(source.id)} value={String(source.id)}>
                        {String(source.name || source.id)}
                      </option>
                    ))}
                  </select>
                  <div className="grid grid-cols-2 gap-3">
                    <select
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={String(codingSwarmProfileDraft.preset_key || 'bug_triage_swarm')}
                      onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, preset_key: e.target.value }))}
                    >
                      <option value="bug_triage_swarm">Bug Triage Swarm</option>
                      <option value="build_break_swarm">Build Break Swarm</option>
                      <option value="frontend_regression_swarm">Frontend Regression Swarm</option>
                    </select>
                    <select
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={String(codingSwarmProfileDraft.scope_default || 'auto')}
                      onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, scope_default: e.target.value }))}
                    >
                      <option value="auto">Auto scope</option>
                      <option value="backend">Backend</option>
                      <option value="frontend">Frontend</option>
                      <option value="worker">Worker</option>
                    </select>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <input
                      type="number"
                      min={1}
                      max={4}
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={Number(codingSwarmProfileDraft.max_agents || 4)}
                      onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, max_agents: Math.max(1, Math.min(parseInt(e.target.value || '4', 10), 4)) }))}
                    />
                    <select
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={String(codingSwarmProfileDraft.safe_command_policy || 'standard')}
                      onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, safe_command_policy: e.target.value }))}
                    >
                      <option value="standard">Standard</option>
                    </select>
                  </div>
                  <input
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    placeholder="Saved search query"
                    value={String(codingSwarmProfileDraft.saved_search_query || '')}
                    onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, saved_search_query: e.target.value }))}
                  />
                  <textarea
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                    rows={3}
                    placeholder="Default commands, one per line"
                    value={Array.isArray(codingSwarmProfileDraft.default_commands) ? codingSwarmProfileDraft.default_commands.join('\n') : ''}
                    onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, default_commands: parseQuickStartCommands(e.target.value, 8) }))}
                  />
                  <textarea
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                    rows={3}
                    placeholder="Default file paths, one per line"
                    value={Array.isArray(codingSwarmProfileDraft.default_file_paths) ? codingSwarmProfileDraft.default_file_paths.join('\n') : ''}
                    onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, default_file_paths: parseSafeRelativeFilePaths(e.target.value, 16).items }))}
                  />
                  <div className="flex items-center justify-between gap-3">
                    <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        className="rounded border-gray-300"
                        checked={Boolean(codingSwarmProfileDraft.is_default)}
                        onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, is_default: e.target.checked }))}
                      />
                      Make default
                    </label>
                    <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        className="rounded border-gray-300"
                        checked={String(codingSwarmProfileDraft.status || 'active').toLowerCase() === 'active'}
                        onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, status: e.target.checked ? 'active' : 'disabled' }))}
                      />
                      Active
                    </label>
                  </div>
                  <div className="space-y-2 border border-gray-200 rounded-lg p-3">
                    <div className="text-sm font-medium text-gray-700">Sharing</div>
                    <select
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={String(codingSwarmProfileDraft.visibility || 'private')}
                      onChange={(e) => setCodingSwarmProfileDraft((prev) => ({ ...prev, visibility: e.target.value }))}
                    >
                      <option value="private">Private</option>
                      <option value="shared">Shared</option>
                    </select>
                    {String(codingSwarmProfileDraft.visibility || 'private') === 'shared' ? (
                      <div className="grid grid-cols-1 gap-1 max-h-40 overflow-y-auto">
                        {collaborationUsers
                          .filter((candidate) => String(candidate.id) !== String(user?.id || ''))
                          .map((candidate) => {
                            const candidateId = String(candidate.id);
                            const checked = Array.isArray(codingSwarmProfileDraft.shared_with_user_ids) && codingSwarmProfileDraft.shared_with_user_ids.includes(candidateId);
                            return (
                              <label key={candidateId} className="inline-flex items-center gap-2 text-sm text-gray-700">
                                <input
                                  type="checkbox"
                                  className="rounded border-gray-300"
                                  checked={checked}
                                  onChange={(e) =>
                                    setCodingSwarmProfileDraft((prev) => {
                                      const current = Array.isArray(prev.shared_with_user_ids) ? prev.shared_with_user_ids : [];
                                      return {
                                        ...prev,
                                        shared_with_user_ids: e.target.checked
                                          ? Array.from(new Set([...current, candidateId]))
                                          : current.filter((value) => value !== candidateId),
                                      };
                                    })
                                  }
                                />
                                {userLabelById(candidateId)}
                              </label>
                            );
                          })}
                      </div>
                    ) : null}
                  </div>
                  <div className="flex gap-2 pt-2">
                    <Button
                      variant="primary"
                      disabled={
                        (!codingSwarmProfileDraft.title || !String(codingSwarmProfileDraft.source_id || '').trim()) ||
                        createCodingSwarmProfileMutation.isLoading ||
                        updateCodingSwarmProfileMutation.isLoading
                      }
                      onClick={async () => {
                        const payload = {
                          title: String(codingSwarmProfileDraft.title || '').trim(),
                          source_id: String(codingSwarmProfileDraft.source_id || '').trim(),
                          preset_key: String(codingSwarmProfileDraft.preset_key || 'bug_triage_swarm').trim(),
                          description: String(codingSwarmProfileDraft.description || '').trim() || undefined,
                          scope_default: String(codingSwarmProfileDraft.scope_default || 'auto').trim() || 'auto',
                          default_commands: Array.isArray(codingSwarmProfileDraft.default_commands) ? codingSwarmProfileDraft.default_commands : [],
                          default_file_paths: Array.isArray(codingSwarmProfileDraft.default_file_paths) ? codingSwarmProfileDraft.default_file_paths : [],
                          max_agents: Math.max(1, Math.min(Number(codingSwarmProfileDraft.max_agents || 4), 4)),
                          safe_command_policy: String(codingSwarmProfileDraft.safe_command_policy || 'standard').trim() || 'standard',
                          saved_search_query: String(codingSwarmProfileDraft.saved_search_query || '').trim() || undefined,
                          is_default: Boolean(codingSwarmProfileDraft.is_default),
                          status: String(codingSwarmProfileDraft.status || 'active').trim() || 'active',
                          visibility: String(codingSwarmProfileDraft.visibility || 'private').trim() || 'private',
                          shared_with_user_ids: Array.isArray(codingSwarmProfileDraft.shared_with_user_ids) ? codingSwarmProfileDraft.shared_with_user_ids : [],
                        };
                        if (editingCodingSwarmProfileId && !codingSwarmProfileDraft.duplicate_mode) {
                          await updateCodingSwarmProfileMutation.mutateAsync({
                            profileId: editingCodingSwarmProfileId,
                            data: {
                              title: payload.title,
                              description: payload.description,
                              preset_key: payload.preset_key,
                              scope_default: payload.scope_default,
                              default_commands: payload.default_commands,
                              default_file_paths: payload.default_file_paths,
                              max_agents: payload.max_agents,
                              safe_command_policy: payload.safe_command_policy,
                              saved_search_query: payload.saved_search_query,
                              is_default: payload.is_default,
                              status: payload.status,
                              visibility: payload.visibility,
                              shared_with_user_ids: payload.shared_with_user_ids,
                            },
                          });
                        } else {
                          await createCodingSwarmProfileMutation.mutateAsync(payload);
                        }
                        closeCodingSwarmProfileEditor();
                      }}
                    >
                      {editingCodingSwarmProfileId && !codingSwarmProfileDraft.duplicate_mode ? 'Save profile' : 'Create profile'}
                    </Button>
                    <Button variant="secondary" onClick={closeCodingSwarmProfileEditor}>
                      Cancel
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'backlog' && (
          <div className="w-full flex flex-col min-h-0 gap-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-1 bg-white border border-gray-200 rounded-lg p-4 space-y-3">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">Coding Backlog</h2>
                  <p className="text-sm text-gray-500">Curated portfolio goals that spawn bounded repo repair/apply jobs.</p>
                </div>
                <input
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  placeholder="Backlog title"
                  value={backlogTitle}
                  onChange={(e) => setBacklogTitle(e.target.value)}
                />
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={4}
                  placeholder="Portfolio goal"
                  value={backlogGoal}
                  onChange={(e) => setBacklogGoal(e.target.value)}
                />
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={3}
                  placeholder="Observed failure symptom (optional but recommended)"
                  value={backlogFailureSymptom}
                  onChange={(e) => setBacklogFailureSymptom(e.target.value)}
                />
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={backlogSourceId}
                  onChange={(e) => setBacklogSourceId(e.target.value)}
                >
                  <option value="">Select repo source</option>
                  {codeSources.map((source: any) => (
                    <option key={String(source.id)} value={String(source.id)}>
                      {String(source.name || source.id)}
                    </option>
                  ))}
                </select>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={backlogScope}
                  onChange={(e) => setBacklogScope(e.target.value)}
                >
                  <option value="auto">Auto scope</option>
                  <option value="backend">Backend</option>
                  <option value="frontend">Frontend</option>
                  <option value="worker">Worker</option>
                </select>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={2}
                  placeholder="Verification commands, one per line (optional)"
                  value={backlogCommandsText}
                  onChange={(e) => setBacklogCommandsText(e.target.value)}
                />
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={2}
                  placeholder="File path hints, one per line (optional)"
                  value={backlogFilePathsText}
                  onChange={(e) => setBacklogFilePathsText(e.target.value)}
                />
                <div className="flex gap-2">
                  <Button
                    variant="primary"
                    disabled={createCodingBacklogMutation.isLoading || !backlogTitle.trim() || !backlogGoal.trim() || !backlogSourceId}
                    onClick={() =>
                      createCodingBacklogMutation.mutate({
                        title: backlogTitle.trim(),
                        portfolio_goal: backlogGoal.trim(),
                        source_id: backlogSourceId,
                        scope: backlogScope,
                        failure_symptom: backlogFailureSymptom.trim() || undefined,
                        commands: backlogCommandsText.split('\n').map((v) => v.trim()).filter(Boolean),
                        file_paths: backlogFilePathsText.split('\n').map((v) => v.trim()).filter(Boolean),
                        auto_apply_enabled: true,
                        require_patch_pr: false,
                        policy: { max_auto_retries: 1 },
                        start_immediately: true,
                      })
                    }
                  >
                    Start Backlog
                  </Button>
                  <Button variant="ghost" onClick={() => refetchCodingBacklog()}>
                    <RefreshCw className="w-4 h-4" />
                  </Button>
                </div>
              </div>
              <div className="col-span-2 bg-white border border-gray-200 rounded-lg p-4 min-h-0">
                <div className="flex flex-wrap gap-3 items-center mb-4">
                  <select
                    className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={backlogVisibilityScope}
                    onChange={(e) => setBacklogVisibilityScope(e.target.value as 'mine' | 'shared' | 'all')}
                  >
                    <option value="mine">My backlog</option>
                    <option value="shared">Shared with me</option>
                    <option value="all">All visible</option>
                  </select>
                  <select
                    className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={backlogAssignmentFilter}
                    onChange={(e) => setBacklogAssignmentFilter(e.target.value)}
                  >
                    <option value="">Any assignee</option>
                    <option value={String(user?.id || '')}>Assigned to me</option>
                    {collaborationUsers.map((candidate) => (
                      <option key={String(candidate.id)} value={String(candidate.id)}>
                        {userLabelById(String(candidate.id))}
                      </option>
                    ))}
                  </select>
                  <select
                    className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={backlogQueueStateFilter}
                    onChange={(e) => setBacklogQueueStateFilter(e.target.value)}
                  >
                    <option value="">Any queue state</option>
                    <option value="new_auto_routed">New auto-routed</option>
                    <option value="awaiting_assignment">Awaiting assignment</option>
                    <option value="ready_to_start">Ready to start</option>
                    <option value="awaiting_operator_decision">Awaiting operator decision</option>
                    <option value="in_progress">In progress</option>
                    <option value="blocked">Blocked</option>
                    <option value="superseded">Superseded</option>
                  </select>
                  {(backlogVisibilityScope !== 'mine' || backlogAssignmentFilter || backlogQueueStateFilter) ? (
                    <Button variant="ghost" size="sm" onClick={() => { setBacklogVisibilityScope('mine'); setBacklogAssignmentFilter(''); setBacklogQueueStateFilter(''); }}>
                      <XCircle className="w-4 h-4 mr-1" />
                      Clear
                    </Button>
                  ) : null}
                </div>
                {codingBacklogLoading ? (
                  <div className="flex justify-center items-center h-48"><LoadingSpinner /></div>
                ) : (
                  <div className="space-y-3">
                    {filteredBacklogItems.map((item) => {
                      const summary = (item.latest_summary || {}) as CodingBacklogLatestSummary;
                      const decomposition = (item.decomposition || {
                        planned_slices: [],
                        completed_slices: [],
                        failed_slices: [],
                        promotion_decisions: [],
                        portfolio_progress: null,
                      }) as CodingBacklogDecomposition;
                      const plannedSlices = Array.isArray(decomposition.planned_slices) ? decomposition.planned_slices : [];
                      const promotionDecisions = Array.isArray(decomposition.promotion_decisions) ? decomposition.promotion_decisions : [];
                      const progress = decomposition.portfolio_progress || summary?.portfolio_progress || null;
                      const activeSliceId = String(decomposition.active_slice_id || summary?.active_slice_id || '').trim();
                      const activeSlice = plannedSlices.find((slice) => String(slice?.slice_id || '').trim() === activeSliceId) || null;
                      const policy = (item.policy || {}) as CodingBacklogPolicy;
                      const childCount = Array.isArray(item.child_job_ids) ? item.child_job_ids.length : 0;
                      const backlogLineage = ((item as any)?.lineage && typeof (item as any).lineage === 'object')
                        ? ((item as any).lineage as Record<string, any>)
                        : {};
                      const originatingSwarmJobId = String(backlogLineage.originating_swarm_job_id || '').trim();
                      const originatingSwarmPreset = String(backlogLineage.originating_swarm_preset || '').trim();
                      const originatingSwarmReviewReason = String(backlogLineage.originating_swarm_review_reason || '').trim();
                      const originatingSwarmRouteMode = String(backlogLineage.originating_swarm_route_mode || '').trim().toLowerCase();
                      const originatingSwarmOutcome = originatingSwarmJobId ? swarmOutcomeBySwarmJobId[originatingSwarmJobId] || null : null;
                      const queueState = String(item.operator_queue_state || '').trim();
                      const whyNotRepair = ((item.why_not_repair && typeof item.why_not_repair === 'object') ? item.why_not_repair : {}) as Record<string, any>;
                      const operatorNote = String((item.collaboration as any)?.note || summary?.operator_note || '').trim();
                      const collaborationSummary = ((item.collaboration_summary && typeof item.collaboration_summary === 'object')
                        ? item.collaboration_summary
                        : {}) as Record<string, any>;
                      const backlogNoteValue = backlogNoteDrafts[String(item.id)] ?? operatorNote;
                      const backlogCloseReasonValue = backlogCloseReasonDrafts[String(item.id)] ?? String(item.closure_reason || '');
                      const chipBase = 'text-xs px-2 py-0.5 rounded';
                      const backlogWaiting = Boolean(summary?.waiting_on_operator_action);
                      return (
                        <div key={item.id} className="border border-gray-200 rounded-lg p-4">
                          <div className="flex items-start justify-between gap-4">
                            <div className="min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <h3 className="section-heading">{item.title}</h3>
                                <span className={`${chipBase} bg-slate-100 text-slate-700`}>{item.status}</span>
                                <span className={`${chipBase} bg-blue-100 text-blue-700`}>Priority {item.priority}</span>
                                {summary?.promotion_decision ? (
                                  <span className={`${chipBase} ${String(summary.promotion_decision) === 'auto_applied' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'}`}>
                                    {String(summary.promotion_decision).replace(/_/g, ' ')}
                                  </span>
                                ) : null}
                                {activeSlice ? (
                                  <span className={`${chipBase} bg-violet-100 text-violet-700`}>
                                    Active {String(activeSlice.status || 'pending').replace(/_/g, ' ')}
                                  </span>
                                ) : null}
                                {originatingSwarmPreset ? (
                                  <span className={`${chipBase} bg-rose-50 text-rose-700 border border-rose-100`}>
                                    From {originatingSwarmPreset.replace(/_/g, ' ')}
                                  </span>
                                ) : null}
                                {originatingSwarmRouteMode ? (
                                  <span className={`${chipBase} ${originatingSwarmRouteMode === 'auto' ? 'bg-amber-50 text-amber-700 border border-amber-100' : 'bg-slate-100 text-slate-700'}`}>
                                    {originatingSwarmRouteMode === 'auto' ? 'Auto-routed' : 'Manual backlog'}
                                  </span>
                                ) : null}
                                {queueState ? (
                                  <span className={`${chipBase} bg-cyan-50 text-cyan-700 border border-cyan-100`}>
                                    {queueState.replace(/_/g, ' ')}
                                  </span>
                                ) : null}
                                {item.closure_reason ? (
                                  <span className={`${chipBase} bg-slate-100 text-slate-700 border border-slate-200`}>
                                    {String(item.closure_reason).replace(/_/g, ' ')}
                                  </span>
                                ) : null}
                              </div>
                              <div className="text-sm text-gray-600 whitespace-pre-wrap">{item.portfolio_goal}</div>
                              {item.failure_symptom ? (
                                <div className="text-xs text-amber-700 mt-2">Symptom: {item.failure_symptom}</div>
                              ) : null}
                              <CollaborationSummaryPanel
                                summary={collaborationSummary as CollaborationSummary}
                                fallbackOwnerId={String((item as any).collaboration?.owner_user_id || item.user_id || '')}
                                fallbackVisibility={String(collaborationSummary.visibility_scope || item.visibility || 'private')}
                                fallbackSharedWithUserIds={Array.isArray(collaborationSummary.shared_with_user_ids || item.shared_with_user_ids) ? [...(collaborationSummary.shared_with_user_ids || item.shared_with_user_ids || [])].map((value) => String(value || '').trim()).filter(Boolean) : []}
                                userLabelById={userLabelById}
                                assigneeUsers={collaborationUsers}
                                showAssigneeSelect
                                assigneeValue={String(item.assigned_user_id || '')}
                                onAssigneeChange={(nextAssignee) => {
                                  if (!nextAssignee) {
                                    codingBacklogActionMutation.mutate({ itemId: item.id, action: 'clear_backlog_assignment' });
                                  } else {
                                    codingBacklogActionMutation.mutate({ itemId: item.id, action: 'assign_backlog', assignedUserId: nextAssignee });
                                  }
                                }}
                                onClearAssignee={() => codingBacklogActionMutation.mutate({ itemId: item.id, action: 'clear_backlog_assignment' })}
                                noteValue={backlogNoteValue}
                                onNoteChange={(value) =>
                                  setBacklogNoteDrafts((prev) => ({
                                    ...prev,
                                    [String(item.id)]: value,
                                  }))
                                }
                                onNoteSave={() =>
                                  codingBacklogActionMutation.mutate({
                                    itemId: item.id,
                                    action: 'update_backlog_note',
                                    operatorNote: backlogNoteValue,
                                  })
                                }
                                noteSaveLabel="Save note"
                                notePlaceholder="Backlog operator note"
                              />
                              <div className="text-xs text-gray-500 mt-2 flex flex-wrap gap-3">
                                <span>Child jobs: {childCount}</span>
                                {item.current_job_id ? <span>Current job: {item.current_job_id}</span> : null}
                                {summary?.promotion_decision ? <span>Promotion: {String(summary.promotion_decision)}</span> : null}
                                {progress ? <span>Slices {Number(progress.completed_slices || 0)}/{Number(progress.total_slices || 0)}</span> : null}
                                {originatingSwarmJobId ? <span>Swarm job: {originatingSwarmJobId}</span> : null}
                                {String(backlogLineage.originating_swarm_candidate_role || '').trim() ? (
                                  <span>Candidate role: {String(backlogLineage.originating_swarm_candidate_role)}</span>
                                ) : null}
                              </div>
                              {originatingSwarmReviewReason ? (
                                <div className="text-xs text-gray-500 mt-2">Swarm review: {originatingSwarmReviewReason}</div>
                              ) : null}
                              {originatingSwarmJobId ? (
                                <div className="mt-3 rounded border border-cyan-100 bg-cyan-50 p-3 text-xs text-cyan-900">
                                  <div className="font-medium">Swarm triage summary</div>
                                  <div className="mt-1">
                                    Why not repair: {String(whyNotRepair.review_reason || originatingSwarmReviewReason || 'Insufficient swarm consensus').trim()}
                                  </div>
                                  <div className="mt-1 flex flex-wrap gap-3">
                                    {whyNotRepair.candidate_role ? <span>Candidate role {String(whyNotRepair.candidate_role)}</span> : null}
                                    {whyNotRepair.route_mode ? <span>Route {String(whyNotRepair.route_mode)}</span> : null}
                                    {whyNotRepair.recommended_next_action ? <span>Suggested {String(whyNotRepair.recommended_next_action).replace(/_/g, ' ')}</span> : null}
                                  </div>
                                  {Array.isArray(item.file_paths) && item.file_paths.length > 0 ? (
                                    <div className="mt-1">Files: {item.file_paths.slice(0, 4).join(', ')}</div>
                                  ) : null}
                                  {Array.isArray(item.commands) && item.commands.length > 0 ? (
                                    <div className="mt-1">Commands: {item.commands.slice(0, 2).join(' | ')}</div>
                                  ) : null}
                                </div>
                              ) : null}
                              {originatingSwarmOutcome ? (
                                <div className="text-xs text-gray-500 mt-2">
                                  Swarm outcome: {humanizeSwarmOutcome(originatingSwarmOutcome.terminal_outcome)}
                                  {originatingSwarmOutcome.repair_job_id ? ` · Repair ${String(originatingSwarmOutcome.repair_status || 'linked')}` : ''}
                                  {originatingSwarmOutcome.verification_status ? ` · Verification ${humanizeSwarmOutcome(originatingSwarmOutcome.verification_status)}` : ''}
                                </div>
                              ) : null}
                              {summary?.blocked_reason ? (
                                <div className="text-xs text-rose-700 mt-2">Blocked: {String(summary.blocked_reason)}</div>
                              ) : null}
                              {summary?.note ? (
                                <div className="text-xs text-gray-500 mt-2">{String(summary.note)}</div>
                              ) : null}
                              {operatorNote ? (
                                <div className="text-xs text-slate-600 mt-2">Operator note: {operatorNote}</div>
                              ) : null}
                            </div>
                            <div className="flex gap-2 shrink-0">
                              {originatingSwarmJobId ? (
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => {
                                    setSelectedJob(null);
                                    navigate(buildAutonomousAgentsUrl(originatingSwarmJobId));
                                    setActiveTab('jobs');
                                  }}
                                >
                                  Open swarm
                                </Button>
                              ) : null}
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => codingBacklogActionMutation.mutate({ itemId: item.id, action: 'assign_backlog', assignedUserId: String(user?.id || '') })}
                              >
                                Assign to me
                              </Button>
                              <select
                                className="border border-gray-300 rounded-lg px-2 py-1 text-xs"
                                value={backlogCloseReasonValue}
                                onChange={(e) =>
                                  setBacklogCloseReasonDrafts((prev) => ({
                                    ...prev,
                                    [String(item.id)]: e.target.value,
                                  }))
                                }
                              >
                                <option value="">Choose close reason</option>
                                <option value="duplicate">Duplicate</option>
                                <option value="false_alarm">False alarm</option>
                                <option value="outdated">Outdated</option>
                                <option value="blocked_external">Blocked external</option>
                                <option value="fixed_through_backlog">Fixed through backlog</option>
                                <option value="promoted_to_repair">Promoted to repair</option>
                              </select>
                              {['draft', 'failed', 'completed'].includes(item.status) ? (
                                <Button
                                  size="sm"
                                  variant="primary"
                                  onClick={() => codingBacklogActionMutation.mutate({ itemId: item.id, action: 'start' })}
                                >
                                  Start
                                </Button>
                              ) : null}
                              {item.status === 'running' ? (
                                <Button
                                  size="sm"
                                  variant="secondary"
                                  onClick={() => codingBacklogActionMutation.mutate({ itemId: item.id, action: 'pause' })}
                                >
                                  Pause
                                </Button>
                              ) : null}
                              {item.status === 'paused' ? (
                                <Button
                                  size="sm"
                                  variant="secondary"
                                  onClick={() => codingBacklogActionMutation.mutate({ itemId: item.id, action: 'resume' })}
                                >
                                  Resume
                                </Button>
                              ) : null}
                              {['running', 'paused', 'draft', 'failed'].includes(item.status) ? (
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => {
                                    const closureReason = String(backlogCloseReasonDrafts[String(item.id)] || '').trim();
                                    if (!closureReason) {
                                      toast.error('Choose a close reason');
                                      return;
                                    }
                                    codingBacklogActionMutation.mutate({
                                      itemId: item.id,
                                      action: 'cancel',
                                      closureReason,
                                      operatorNote: backlogNoteValue || undefined,
                                    });
                                  }}
                                  disabled={!String(backlogCloseReasonValue || '').trim()}
                                >
                                  Close
                                </Button>
                              ) : null}
                            </div>
                          </div>
                          {backlogWaiting ? (
                            <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                              Awaiting operator decision
                              {summary?.recommended_next_action ? ` · Recommended ${String(summary.recommended_next_action).replace(/_/g, ' ')}` : ''}
                            </div>
                          ) : null}
                          <details className="mt-3 bg-gray-50 border border-gray-100 rounded-lg p-3">
                            <summary className="cursor-pointer text-xs font-medium text-gray-800">Orchestration detail</summary>
                            <div className="mt-3 space-y-3 text-xs text-gray-700">
                              {progress ? (
                                <div className="grid grid-cols-3 gap-2">
                                  <div className="bg-white border border-gray-200 rounded p-2">
                                    <div className="text-gray-500">Portfolio progress</div>
                                    <div className="mt-1 font-medium text-gray-900">
                                      {Number(progress.completed_slices || 0)}/{Number(progress.total_slices || 0)} completed
                                    </div>
                                    <div className="text-gray-500">
                                      Pending {Number(progress.pending_slices || 0)} · Failed {Number(progress.failed_slices || 0)}
                                    </div>
                                  </div>
                                  <div className="bg-white border border-gray-200 rounded p-2">
                                    <div className="text-gray-500">Promotion outcomes</div>
                                    <div className="mt-1 font-medium text-gray-900">
                                      Auto-applied {Number(progress.auto_applied_slices || 0)}
                                    </div>
                                    <div className="text-gray-500">Proposal-only {Number(progress.proposal_only_slices || 0)}</div>
                                  </div>
                                  <div className="bg-white border border-gray-200 rounded p-2">
                                    <div className="text-gray-500">Auto-apply policy</div>
                                    <div className="mt-1 text-gray-600">
                                      Max files {Number(policy.max_files_touched || 0)} · Retries {Number(policy.max_auto_retries || 0)}
                                    </div>
                                    <div className="text-gray-500">
                                      Confidence {typeof policy.confidence_threshold === 'number' ? policy.confidence_threshold.toFixed(2) : 'n/a'}
                                    </div>
                                  </div>
                                </div>
                              ) : null}

                              {summary?.promotion_evaluation ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Latest promotion evaluation</div>
                                  <div className="mt-1 text-gray-600">
                                    Decision {String((summary.promotion_evaluation as Record<string, any>).decision || summary.promotion_decision || 'proposal_only').replace(/_/g, ' ')}
                                  </div>
                                  <div className="text-gray-500">
                                    Confidence {Number((summary.promotion_evaluation as Record<string, any>).proposal_confidence || 0).toFixed(2)}
                                    {' '}· Files {Number((summary.promotion_evaluation as Record<string, any>).files_touched_count || 0)}
                                    {' '}· Verified {String((summary.promotion_evaluation as Record<string, any>).experiment_ok)}
                                  </div>
                                  {(summary.promotion_evaluation as Record<string, any>).blocked_reason ? (
                                    <div className="text-rose-700 mt-1">
                                      Blocked: {String((summary.promotion_evaluation as Record<string, any>).blocked_reason).replace(/_/g, ' ')}
                                    </div>
                                  ) : null}
                                </div>
                              ) : null}

                              {plannedSlices.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2 space-y-2">
                                  <div className="font-medium text-gray-800">Planned slices</div>
                                  {plannedSlices.map((slice: CodingBacklogSlice) => {
                                    const isActive = String(slice.slice_id || '') === activeSliceId;
                                    return (
                                      <div key={String(slice.slice_id)} className="border border-gray-100 rounded p-2">
                                        <div className="flex items-center justify-between gap-2">
                                          <div className="font-medium text-gray-800">{String(slice.title || slice.slice_id)}</div>
                                          <div className="flex gap-1 flex-wrap justify-end">
                                            <span className={`${chipBase} ${isActive ? 'bg-violet-100 text-violet-700' : 'bg-slate-100 text-slate-700'}`}>
                                              {String(slice.status || 'pending').replace(/_/g, ' ')}
                                            </span>
                                            {slice.promotion_decision ? (
                                              <span className={`${chipBase} ${String(slice.promotion_decision) === 'auto_applied' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'}`}>
                                                {String(slice.promotion_decision).replace(/_/g, ' ')}
                                              </span>
                                            ) : null}
                                          </div>
                                        </div>
                                        <div className="mt-1 text-gray-500">
                                          Scope {String(slice.scope || 'auto')} · Retries {Number(slice.retry_count || 0)}
                                          {slice.proposal_confidence ? ` · Confidence ${Number(slice.proposal_confidence || 0).toFixed(2)}` : ''}
                                        </div>
                                        {Array.isArray(slice.file_paths) && slice.file_paths.length > 0 ? (
                                          <div className="mt-1 text-gray-600 font-mono break-all">
                                            {slice.file_paths.join('\n')}
                                          </div>
                                        ) : null}
                                        {slice.blocked_reason ? (
                                          <div className="mt-1 text-rose-700">Blocked: {String(slice.blocked_reason).replace(/_/g, ' ')}</div>
                                        ) : null}
                                        {slice.awaiting_operator_action && Array.isArray(slice.allowed_slice_actions) && slice.allowed_slice_actions.length > 0 ? (
                                          <div className="mt-2 flex flex-wrap gap-2">
                                            {slice.allowed_slice_actions.includes('apply_override') ? (
                                              <Button
                                                size="sm"
                                                variant="primary"
                                                onClick={() => codingBacklogActionMutation.mutate({ itemId: item.id, action: 'apply_override', sliceId: slice.slice_id })}
                                              >
                                                Apply Override
                                              </Button>
                                            ) : null}
                                            {slice.allowed_slice_actions.includes('create_patch_pr') ? (
                                              <Button
                                                size="sm"
                                                variant="secondary"
                                                onClick={() => codingBacklogActionMutation.mutate({ itemId: item.id, action: 'create_patch_pr', sliceId: slice.slice_id })}
                                              >
                                                Create Patch PR
                                              </Button>
                                            ) : null}
                                            {slice.allowed_slice_actions.includes('keep_proposal_only') ? (
                                              <Button
                                                size="sm"
                                                variant="ghost"
                                                onClick={() => codingBacklogActionMutation.mutate({ itemId: item.id, action: 'keep_proposal_only', sliceId: slice.slice_id })}
                                              >
                                                Keep Proposal
                                              </Button>
                                            ) : null}
                                            {slice.allowed_slice_actions.includes('relaunch_slice') ? (
                                              <Button
                                                size="sm"
                                                variant="secondary"
                                                onClick={() => codingBacklogActionMutation.mutate({ itemId: item.id, action: 'relaunch_slice', sliceId: slice.slice_id })}
                                              >
                                                Relaunch Slice
                                              </Button>
                                            ) : null}
                                            {slice.allowed_slice_actions.includes('skip_slice') ? (
                                              <Button
                                                size="sm"
                                                variant="ghost"
                                                onClick={() => codingBacklogActionMutation.mutate({ itemId: item.id, action: 'skip_slice', sliceId: slice.slice_id })}
                                              >
                                                Skip Slice
                                              </Button>
                                            ) : null}
                                          </div>
                                        ) : null}
                                        {Array.isArray(slice.timeline) && slice.timeline.length > 0 ? (
                                          <details className="mt-2 rounded border border-gray-200 bg-gray-50 p-2">
                                            <summary className="cursor-pointer text-gray-700">Slice timeline</summary>
                                            <div className="mt-2 space-y-1 text-gray-600">
                                              {slice.timeline.map((entry: CodingBacklogTimelineEntry, idx: number) => (
                                                <div key={`${String(entry.action || 'entry')}-${idx}`} className="flex items-start justify-between gap-2">
                                                  <div>
                                                    {entry.at ? `${new Date(String(entry.at)).toLocaleString()} · ` : ''}
                                                    {String(entry.actor || 'system')}
                                                    {' '}· {String(entry.action || 'state_change').replace(/_/g, ' ')}
                                                    {entry.job_id ? ` · Job ${String(entry.job_id)}` : ''}
                                                    {entry.patch_pr_id ? ` · Patch PR ${String(entry.patch_pr_id)}` : ''}
                                                    {entry.note ? ` · ${String(entry.note)}` : ''}
                                                  </div>
                                                  <div className="flex gap-2 shrink-0">
                                                    {entry.job_id ? (
                                                      <Button size="sm" variant="ghost" onClick={() => openBacklogJob(String(entry.job_id))}>
                                                        Open Job
                                                      </Button>
                                                    ) : null}
                                                    {entry.patch_pr_id ? (
                                                      <Button size="sm" variant="ghost" onClick={() => openPatchPr(String(entry.patch_pr_id))}>
                                                        Open Patch PR
                                                      </Button>
                                                    ) : null}
                                                  </div>
                                                </div>
                                              ))}
                                            </div>
                                          </details>
                                        ) : null}
                                        {((slice.job_lineage && (
                                          (Array.isArray(slice.job_lineage.repair_job_ids) && slice.job_lineage.repair_job_ids.length > 0) ||
                                          (Array.isArray(slice.job_lineage.apply_job_ids) && slice.job_lineage.apply_job_ids.length > 0) ||
                                          (Array.isArray(slice.job_lineage.patch_pr_ids) && slice.job_lineage.patch_pr_ids.length > 0) ||
                                          (Array.isArray(slice.job_lineage.proposal_ids) && slice.job_lineage.proposal_ids.length > 0) ||
                                          (Array.isArray(slice.job_lineage.retry_from_job_ids) && slice.job_lineage.retry_from_job_ids.length > 0)
                                        )) || (Array.isArray(slice.artifact_history) && slice.artifact_history.length > 0) || (Array.isArray(slice.manual_promotion_history) && slice.manual_promotion_history.length > 0)) ? (
                                          <details className="mt-2 rounded border border-gray-200 bg-gray-50 p-2">
                                            <summary className="cursor-pointer text-gray-700">Artifacts and lineage</summary>
                                            <div className="mt-2 space-y-2 text-gray-600">
                                              {slice.job_lineage ? (
                                                <div>
                                                  <div className="font-medium text-gray-800">Job lineage</div>
                                                  {Array.isArray(slice.job_lineage.repair_job_ids) && slice.job_lineage.repair_job_ids.length > 0 ? (
                                                    <div>
                                                      <div>Repair jobs: {slice.job_lineage.repair_job_ids.join(', ')}</div>
                                                      <div className="mt-1 flex flex-wrap gap-2">
                                                        {slice.job_lineage.repair_job_ids.map((jobId) => (
                                                          <Button key={`repair-${jobId}`} size="sm" variant="ghost" onClick={() => openBacklogJob(String(jobId))}>
                                                            Open {String(jobId)}
                                                          </Button>
                                                        ))}
                                                      </div>
                                                    </div>
                                                  ) : null}
                                                  {Array.isArray(slice.job_lineage.apply_job_ids) && slice.job_lineage.apply_job_ids.length > 0 ? (
                                                    <div>
                                                      <div>Apply jobs: {slice.job_lineage.apply_job_ids.join(', ')}</div>
                                                      <div className="mt-1 flex flex-wrap gap-2">
                                                        {slice.job_lineage.apply_job_ids.map((jobId) => (
                                                          <Button key={`apply-${jobId}`} size="sm" variant="ghost" onClick={() => openBacklogJob(String(jobId))}>
                                                            Open {String(jobId)}
                                                          </Button>
                                                        ))}
                                                      </div>
                                                    </div>
                                                  ) : null}
                                                  {Array.isArray(slice.job_lineage.patch_pr_ids) && slice.job_lineage.patch_pr_ids.length > 0 ? (
                                                    <div>
                                                      <div>Patch PRs: {slice.job_lineage.patch_pr_ids.join(', ')}</div>
                                                      <div className="mt-1 flex flex-wrap gap-2">
                                                        {slice.job_lineage.patch_pr_ids.map((patchPrId) => (
                                                          <Button key={`patch-pr-${patchPrId}`} size="sm" variant="ghost" onClick={() => openPatchPr(String(patchPrId))}>
                                                            Open {String(patchPrId)}
                                                          </Button>
                                                        ))}
                                                      </div>
                                                    </div>
                                                  ) : null}
                                                  {Array.isArray(slice.job_lineage.proposal_ids) && slice.job_lineage.proposal_ids.length > 0 ? (
                                                    <div>
                                                      <div>Proposals: {slice.job_lineage.proposal_ids.join(', ')}</div>
                                                      <div className="mt-1 flex flex-wrap gap-2">
                                                        {slice.job_lineage.proposal_ids.map((proposalId) => (
                                                          <React.Fragment key={`proposal-${proposalId}`}>
                                                            <Button size="sm" variant="ghost" onClick={() => downloadBacklogProposal(String(proposalId), 'Code Patch Proposal')}>
                                                              Download {String(proposalId)}
                                                            </Button>
                                                            <Button size="sm" variant="ghost" onClick={() => copyText(String(proposalId), 'Proposal ID')}>
                                                              Copy ID
                                                            </Button>
                                                          </React.Fragment>
                                                        ))}
                                                      </div>
                                                    </div>
                                                  ) : null}
                                                  {Array.isArray(slice.job_lineage.retry_from_job_ids) && slice.job_lineage.retry_from_job_ids.length > 0 ? (
                                                    <div>
                                                      <div>Retried from: {slice.job_lineage.retry_from_job_ids.join(', ')}</div>
                                                      <div className="mt-1 flex flex-wrap gap-2">
                                                        {slice.job_lineage.retry_from_job_ids.map((jobId) => (
                                                          <Button key={`retry-${jobId}`} size="sm" variant="ghost" onClick={() => openBacklogJob(String(jobId))}>
                                                            Open {String(jobId)}
                                                          </Button>
                                                        ))}
                                                      </div>
                                                    </div>
                                                  ) : null}
                                                </div>
                                              ) : null}
                                              {Array.isArray(slice.artifact_history) && slice.artifact_history.length > 0 ? (
                                                <div>
                                                  <div className="font-medium text-gray-800">Artifacts</div>
                                                  <div className="space-y-1">
                                                    {slice.artifact_history.map((artifact, idx: number) => (
                                                      <div key={`${String(artifact.artifact_type || 'artifact')}-${idx}`} className="flex items-start justify-between gap-2">
                                                        <div>
                                                          {String(artifact.label || artifact.artifact_type || 'artifact')}
                                                          {artifact.artifact_id ? ` · ${String(artifact.artifact_id)}` : ''}
                                                          {artifact.at ? ` · ${new Date(String(artifact.at)).toLocaleString()}` : ''}
                                                        </div>
                                                        <div className="flex gap-2 shrink-0">
                                                          {String(artifact.artifact_type || '') === 'proposal' && artifact.artifact_id ? (
                                                            <>
                                                              <Button size="sm" variant="ghost" onClick={() => downloadBacklogProposal(String(artifact.artifact_id), artifact.label || 'Code Patch Proposal')}>
                                                                Download
                                                              </Button>
                                                              <Button size="sm" variant="ghost" onClick={() => copyText(String(artifact.artifact_id), 'Proposal ID')}>
                                                                Copy ID
                                                              </Button>
                                                            </>
                                                          ) : null}
                                                          {String(artifact.artifact_type || '') === 'patch_pr' && artifact.artifact_id ? (
                                                            <Button size="sm" variant="ghost" onClick={() => openPatchPr(String(artifact.artifact_id))}>
                                                              Open Patch PR
                                                            </Button>
                                                          ) : null}
                                                        </div>
                                                      </div>
                                                    ))}
                                                  </div>
                                                </div>
                                              ) : null}
                                              {Array.isArray(slice.manual_promotion_history) && slice.manual_promotion_history.length > 0 ? (
                                                <div>
                                                  <div className="font-medium text-gray-800">Operator decisions</div>
                                                  <div className="space-y-1">
                                                    {slice.manual_promotion_history.map((event, idx: number) => (
                                                      <div key={`${String(event.action || 'decision')}-${idx}`}>
                                                        {String(event.action || 'decision').replace(/_/g, ' ')}
                                                        {event.at ? ` · ${new Date(String(event.at)).toLocaleString()}` : ''}
                                                        {event.operator_note ? ` · ${String(event.operator_note)}` : ''}
                                                      </div>
                                                    ))}
                                                  </div>
                                                </div>
                                              ) : null}
                                            </div>
                                          </details>
                                        ) : null}
                                      </div>
                                    );
                                  })}
                                </div>
                              ) : null}

                              {promotionDecisions.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2 space-y-1">
                                  <div className="font-medium text-gray-800">Promotion history</div>
                                  {promotionDecisions.map((row, idx) => (
                                    <div key={`${row.slice_id || 'decision'}-${idx}`} className="text-gray-600">
                                      {String(row.title || row.slice_id || 'Slice')}:
                                      {' '}
                                      {String(row.decision || 'proposal_only').replace(/_/g, ' ')}
                                      {row.blocked_reason ? ` (${String(row.blocked_reason).replace(/_/g, ' ')})` : ''}
                                    </div>
                                  ))}
                                </div>
                              ) : null}

                              {Array.isArray(decomposition.backlog_timeline) && decomposition.backlog_timeline.length > 0 ? (
                                <div className="bg-white border border-gray-200 rounded p-2 space-y-1">
                                  <div className="font-medium text-gray-800">Backlog timeline</div>
                                  {decomposition.backlog_timeline.map((entry: CodingBacklogTimelineEntry, idx: number) => (
                                    <div key={`${String(entry.action || 'entry')}-${idx}`} className="flex items-start justify-between gap-2 text-gray-600">
                                      <div>
                                        {entry.at ? `${new Date(String(entry.at)).toLocaleString()} · ` : ''}
                                        {String(entry.actor || 'system')}
                                        {' '}· {String(entry.action || 'state_change').replace(/_/g, ' ')}
                                        {entry.slice_id ? ` · Slice ${String(entry.slice_id)}` : ''}
                                        {entry.job_id ? ` · Job ${String(entry.job_id)}` : ''}
                                        {entry.patch_pr_id ? ` · Patch PR ${String(entry.patch_pr_id)}` : ''}
                                        {entry.note ? ` · ${String(entry.note)}` : ''}
                                      </div>
                                      <div className="flex gap-2 shrink-0">
                                        {entry.job_id ? (
                                          <Button size="sm" variant="ghost" onClick={() => openBacklogJob(String(entry.job_id))}>
                                            Open Job
                                          </Button>
                                        ) : null}
                                        {entry.patch_pr_id ? (
                                          <Button size="sm" variant="ghost" onClick={() => openPatchPr(String(entry.patch_pr_id))}>
                                            Open Patch PR
                                          </Button>
                                        ) : null}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              ) : null}

                              {decomposition.lineage_summary ? (
                                <div className="bg-white border border-gray-200 rounded p-2">
                                  <div className="font-medium text-gray-800">Lineage summary</div>
                                  <div className="mt-1 text-gray-600">
                                    Repair jobs {Number(decomposition.lineage_summary.repair_job_count || 0)}
                                    {' '}· Apply jobs {Number(decomposition.lineage_summary.apply_job_count || 0)}
                                    {' '}· Patch PRs {Number(decomposition.lineage_summary.patch_pr_count || 0)}
                                  </div>
                                  <div className="text-gray-500">
                                    Proposals {Number(decomposition.lineage_summary.proposal_count || 0)}
                                    {' '}· Operator actions {Number(decomposition.lineage_summary.operator_action_count || 0)}
                                  </div>
                                </div>
                              ) : null}
                            </div>
                          </details>
                        </div>
                      );
                    })}
                    {(((codingBacklogData as any)?.items || []) as CodingBacklogItem[]).length === 0 ? (
                      <div className="text-sm text-gray-500">No coding backlog items yet.</div>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        <div className={activeTab === 'jobs' ? 'flex gap-4 flex-1 min-h-0' : 'hidden'}>
            {/* Jobs list */}
            <div className="w-2/3 flex flex-col">
              {/* Filters */}
              <div className="flex gap-3 mb-4">
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                >
                  <option value="">All Status</option>
                  <option value="pending">Pending</option>
                  <option value="running">Running</option>
                  <option value="paused">Paused</option>
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
                  <option value="research">Research</option>
                  <option value="analysis">Analysis</option>
                  <option value="data_analysis">Data Analysis</option>
                  <option value="monitor">Monitor</option>
                  <option value="synthesis">Synthesis</option>
                  <option value="knowledge_expansion">Knowledge Expansion</option>
                  <option value="custom">Custom</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={launchModeFilter}
                  onChange={(e) => setLaunchModeFilter(e.target.value)}
                >
                  <option value="">All launch modes</option>
                  <option value="quick_start_claude_backend">Quick Start: Claude Backend</option>
                  <option value="quick_start_domain_research">Quick Start: Domain Research</option>
                  <option value="quick_start_bug_triage_swarm">Quick Start: Bug Triage Swarm</option>
                  <option value="quick_start_build_break_swarm">Quick Start: Build Break Swarm</option>
                  <option value="quick_start_frontend_regression_swarm">Quick Start: Frontend Regression Swarm</option>
                  <option value="quick_start_repo_bug_triage">Quick Start: Repo Bug Triage</option>
                  <option value="quick_start_role_workflow">Quick Start: Role Workflow</option>
                  <option value="__none__">No launch mode (manual/legacy)</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={hasRelaunchChildrenFilter}
                  onChange={(e) => setHasRelaunchChildrenFilter(e.target.value)}
                >
                  <option value="">Any relaunch children</option>
                  <option value="yes">Has relaunch children</option>
                  <option value="no">No relaunch children</option>
                </select>
                <div className="flex items-center gap-1">
                  <input
                    className={`border rounded-lg px-3 py-2 text-sm w-[220px] ${
                      !isRelaunchFromJobIdFilterValid
                        ? 'border-red-300 bg-red-50'
                        : 'border-gray-300'
                    }`}
                    value={relaunchFromJobIdFilter}
                    onChange={(e) => setRelaunchFromJobIdFilter(String(e.target.value || '').trim())}
                    placeholder="Relaunch parent job id"
                    title="Filter jobs by relaunch parent job id"
                  />
                  {relaunchFromJobIdFilter && (
                    <button
                      type="button"
                      className="text-xs text-gray-500 hover:text-gray-700"
                      onClick={() => setRelaunchFromJobIdFilter('')}
                      title="Clear relaunch parent filter"
                    >
                      clear
                    </button>
                  )}
                </div>
                <label className="inline-flex items-center gap-2 text-sm text-gray-700 px-2">
                  <input
                    type="checkbox"
                    className="rounded border-gray-300"
                    checked={swarmOnlyFilter}
                    onChange={(e) => setSwarmOnlyFilter(Boolean(e.target.checked))}
                  />
                  Swarm only
                </label>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmSortBy}
                  onChange={(e) => setSwarmSortBy(e.target.value)}
                >
                  <option value="created_desc">Newest first</option>
                  <option value="created_asc">Oldest first</option>
                  <option value="swarm_confidence_desc">Swarm confidence</option>
                  <option value="swarm_consensus_desc">Swarm consensus</option>
                  <option value="swarm_conflicts_desc">Swarm conflicts</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={String(swarmMinConsensus)}
                  onChange={(e) => setSwarmMinConsensus(Number(e.target.value || 0))}
                >
                  <option value="0">Any consensus</option>
                  <option value="1">Consensus &ge; 1</option>
                  <option value="2">Consensus &ge; 2</option>
                  <option value="3">Consensus &ge; 3</option>
                  <option value="5">Consensus &ge; 5</option>
                  <option value="8">Consensus &ge; 8</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={graphHealthFilter}
                  onChange={(e) => setGraphHealthFilter(e.target.value)}
                >
                  <option value="">Any graph health</option>
                  <option value="critical">Graph critical</option>
                  <option value="warning">Graph warning</option>
                  <option value="ok">Graph ok</option>
                  <option value="unknown">Graph unknown</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={graphSortBy}
                  onChange={(e) => setGraphSortBy(e.target.value)}
                >
                  <option value="none">Default graph sort</option>
                  <option value="graph_health_critical_first">Graph critical first</option>
                  <option value="graph_severity_desc">Graph severity desc</option>
                  <option value="scope_guard_blocked_first">Scope guard blocked first</option>
                  <option value="experiment_recovery_priority">Experiment recovery priority</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={dedupSkipFilter}
                  onChange={(e) => setDedupSkipFilter(e.target.value)}
                >
                  <option value="">Any dedup skips</option>
                  <option value="gt0">Dedup skipped &gt; 0</option>
                  <option value="gte3">Dedup skipped &ge; 3</option>
                  <option value="gte5">Dedup skipped &ge; 5</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={scopeGuardFilter}
                  onChange={(e) => setScopeGuardFilter(e.target.value)}
                >
                  <option value="">Any scope guards</option>
                  <option value="blocked">Scope guard blocked</option>
                  <option value="clean">No scope guard blocks</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={experimentRecoveryFilter}
                  onChange={(e) => setExperimentRecoveryFilter(e.target.value)}
                >
                  <option value="">Any code recovery</option>
                  <option value="bootstrap_attempted">Bootstrap attempted</option>
                  <option value="bootstrap_recovered">Bootstrap recovered</option>
                  <option value="fallback_attempted">Fallback attempted</option>
                  <option value="fallback_ok">Fallback succeeded</option>
                  <option value="unresolved_recovery">Unresolved fallback recovery</option>
                </select>
                {(launchModeFilter || hasRelaunchChildrenFilter || relaunchFromJobIdFilter || swarmOnlyFilter || swarmSortBy !== 'created_desc' || swarmMinConsensus > 0 || graphHealthFilter || graphSortBy !== 'none' || dedupSkipFilter || scopeGuardFilter || experimentRecoveryFilter) && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setLaunchModeFilter('');
                      setHasRelaunchChildrenFilter('');
                      setRelaunchFromJobIdFilter('');
                      setSwarmOnlyFilter(false);
                      setSwarmSortBy('created_desc');
                      setSwarmMinConsensus(0);
                      setGraphHealthFilter('');
                      setGraphSortBy('none');
                      setDedupSkipFilter('');
                      setScopeGuardFilter('');
                      setExperimentRecoveryFilter('');
                    }}
                    title="Reset swarm and graph filters"
                  >
                    <XCircle className="w-4 h-4 mr-1" />
                    Clear Filters
                  </Button>
                )}
                <Button variant="ghost" size="sm" onClick={() => refetchJobs()}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
              {!isRelaunchFromJobIdFilterValid && (
                <div className="mb-2 text-xs text-red-600">
                  Enter a full job UUID to apply relaunch parent filtering.
                </div>
              )}
              {(hasRelaunchChildrenFilter || relaunchFromJobIdFilter) && (
                <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                  {hasRelaunchChildrenFilter && (
                    <button
                      type="button"
                      className="px-2 py-1 rounded-full border border-cyan-200 bg-cyan-50 text-cyan-800"
                      onClick={() => setHasRelaunchChildrenFilter('')}
                      title="Clear relaunch-children filter"
                    >
                      has children: {hasRelaunchChildrenFilter} ×
                    </button>
                  )}
                  {relaunchFromJobIdFilter && (
                    <button
                      type="button"
                      className="px-2 py-1 rounded-full border border-indigo-200 bg-indigo-50 text-indigo-800 font-mono"
                      onClick={() => setRelaunchFromJobIdFilter('')}
                      title="Clear relaunch parent filter"
                    >
                      parent: {relaunchFromJobIdFilter.slice(0, 12)} ×
                    </button>
                  )}
                </div>
              )}

              <div className="flex flex-wrap items-center gap-2 mb-3 text-xs">
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    !graphHealthFilter
                      ? 'border-primary-200 bg-primary-50 text-primary-700'
                      : 'border-gray-200 bg-gray-50 text-gray-700'
                  }`}
                  onClick={() => setGraphHealthFilter('')}
                  title="Show all graph health statuses"
                >
                  Showing {jobCountSummary.shownCount} / {jobCountSummary.allCount}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    launchModeFilter === 'quick_start_claude_backend'
                      ? 'border-indigo-300 bg-indigo-100 text-indigo-800'
                      : 'border-indigo-100 bg-indigo-50 text-indigo-700'
                  }`}
                  onClick={() =>
                    setLaunchModeFilter((prev) => (prev === 'quick_start_claude_backend' ? '' : 'quick_start_claude_backend'))
                  }
                  title="Toggle filter for Claude backend quick-start jobs"
                >
                  Claude quick-start {jobCountSummary.quick_start_claude_backend}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    launchModeFilter === 'quick_start_domain_research'
                      ? 'border-cyan-300 bg-cyan-100 text-cyan-800'
                      : 'border-cyan-100 bg-cyan-50 text-cyan-700'
                  }`}
                  onClick={() =>
                    setLaunchModeFilter((prev) => (prev === 'quick_start_domain_research' ? '' : 'quick_start_domain_research'))
                  }
                  title="Toggle filter for domain research quick-start jobs"
                >
                  Domain quick-start {jobCountSummary.quick_start_domain_research}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    launchModeFilter === 'quick_start_bug_triage_swarm'
                      ? 'border-rose-300 bg-rose-100 text-rose-800'
                      : 'border-rose-100 bg-rose-50 text-rose-700'
                  }`}
                  onClick={() =>
                    setLaunchModeFilter((prev) => (prev === 'quick_start_bug_triage_swarm' ? '' : 'quick_start_bug_triage_swarm'))
                  }
                  title="Toggle filter for bug triage swarm quick-start jobs"
                >
                  Bug swarm quick-start {jobCountSummary.quick_start_bug_triage_swarm}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    launchModeFilter === 'quick_start_build_break_swarm'
                      ? 'border-amber-300 bg-amber-100 text-amber-800'
                      : 'border-amber-100 bg-amber-50 text-amber-700'
                  }`}
                  onClick={() =>
                    setLaunchModeFilter((prev) => (prev === 'quick_start_build_break_swarm' ? '' : 'quick_start_build_break_swarm'))
                  }
                  title="Toggle filter for build break swarm quick-start jobs"
                >
                  Build swarm quick-start {jobCountSummary.quick_start_build_break_swarm}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    launchModeFilter === 'quick_start_frontend_regression_swarm'
                      ? 'border-cyan-300 bg-cyan-100 text-cyan-800'
                      : 'border-cyan-100 bg-cyan-50 text-cyan-700'
                  }`}
                  onClick={() =>
                    setLaunchModeFilter((prev) =>
                      prev === 'quick_start_frontend_regression_swarm' ? '' : 'quick_start_frontend_regression_swarm'
                    )
                  }
                  title="Toggle filter for frontend regression swarm quick-start jobs"
                >
                  Frontend swarm quick-start {jobCountSummary.quick_start_frontend_regression_swarm}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    launchModeFilter === 'quick_start_repo_bug_triage'
                      ? 'border-amber-300 bg-amber-100 text-amber-800'
                      : 'border-amber-100 bg-amber-50 text-amber-700'
                  }`}
                  onClick={() =>
                    setLaunchModeFilter((prev) => (prev === 'quick_start_repo_bug_triage' ? '' : 'quick_start_repo_bug_triage'))
                  }
                  title="Toggle filter for repo bug triage quick-start jobs"
                >
                  Bug triage quick-start {jobCountSummary.quick_start_repo_bug_triage}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    launchModeFilter === 'quick_start_role_workflow'
                      ? 'border-teal-300 bg-teal-100 text-teal-800'
                      : 'border-teal-100 bg-teal-50 text-teal-700'
                  }`}
                  onClick={() =>
                    setLaunchModeFilter((prev) => (prev === 'quick_start_role_workflow' ? '' : 'quick_start_role_workflow'))
                  }
                  title="Toggle filter for role-workflow quick-start jobs"
                >
                  Role quick-start {jobCountSummary.quick_start_role_workflow}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    graphHealthFilter === 'critical'
                      ? 'border-red-300 bg-red-100 text-red-800'
                      : 'border-red-100 bg-red-50 text-red-700'
                  }`}
                  onClick={() => setGraphHealthFilter('critical')}
                  title="Filter to graph health critical"
                >
                  Critical {jobCountSummary.critical}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    graphHealthFilter === 'warning'
                      ? 'border-amber-300 bg-amber-100 text-amber-800'
                      : 'border-amber-100 bg-amber-50 text-amber-700'
                  }`}
                  onClick={() => setGraphHealthFilter('warning')}
                  title="Filter to graph health warning"
                >
                  Warning {jobCountSummary.warning}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    graphHealthFilter === 'ok'
                      ? 'border-emerald-300 bg-emerald-100 text-emerald-800'
                      : 'border-emerald-100 bg-emerald-50 text-emerald-700'
                  }`}
                  onClick={() => setGraphHealthFilter('ok')}
                  title="Filter to graph health ok"
                >
                  OK {jobCountSummary.ok}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    graphHealthFilter === 'unknown'
                      ? 'border-gray-300 bg-gray-100 text-gray-800'
                      : 'border-gray-100 bg-gray-50 text-gray-600'
                  }`}
                  onClick={() => setGraphHealthFilter('unknown')}
                  title="Filter to graph health unknown"
                >
                  Unknown {jobCountSummary.unknown}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    dedupSkipFilter === 'gt0'
                      ? 'border-indigo-300 bg-indigo-100 text-indigo-800'
                      : 'border-indigo-100 bg-indigo-50 text-indigo-700'
                  }`}
                  onClick={() => setDedupSkipFilter((prev) => (prev === 'gt0' ? '' : 'gt0'))}
                  title="Filter to jobs with any dedup-skipped memories"
                >
                  Dedup&gt;0 {jobCountSummary.dedup_gt0}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    dedupSkipFilter === 'gte3'
                      ? 'border-violet-300 bg-violet-100 text-violet-800'
                      : 'border-violet-100 bg-violet-50 text-violet-700'
                  }`}
                  onClick={() => setDedupSkipFilter((prev) => (prev === 'gte3' ? '' : 'gte3'))}
                  title="Filter to jobs with high dedup-skipped memories"
                >
                  Dedup&ge;3 {jobCountSummary.dedup_gte3}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    scopeGuardFilter === 'blocked'
                      ? 'border-rose-300 bg-rose-100 text-rose-800'
                      : 'border-rose-100 bg-rose-50 text-rose-700'
                  }`}
                  onClick={() => setScopeGuardFilter((prev) => (prev === 'blocked' ? '' : 'blocked'))}
                  title="Filter to jobs with scope guard blocks"
                >
                  Guard blocked {jobCountSummary.scope_guard_blocked}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    scopeGuardFilter === 'clean'
                      ? 'border-sky-300 bg-sky-100 text-sky-800'
                      : 'border-sky-100 bg-sky-50 text-sky-700'
                  }`}
                  onClick={() => setScopeGuardFilter((prev) => (prev === 'clean' ? '' : 'clean'))}
                  title="Filter to jobs without scope guard blocks"
                >
                  Guard clean {Math.max(0, jobCountSummary.shownCount - jobCountSummary.scope_guard_blocked)}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    experimentRecoveryFilter === 'bootstrap_recovered'
                      ? 'border-blue-300 bg-blue-100 text-blue-800'
                      : 'border-blue-100 bg-blue-50 text-blue-700'
                  }`}
                  onClick={() => setExperimentRecoveryFilter((prev) => (prev === 'bootstrap_recovered' ? '' : 'bootstrap_recovered'))}
                  title="Filter to jobs whose latest experiment run recovered after bootstrap"
                >
                  Bootstrap recovered {jobCountSummary.bootstrap_recovered}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    experimentRecoveryFilter === 'fallback_attempted'
                      ? 'border-indigo-300 bg-indigo-100 text-indigo-800'
                      : 'border-indigo-100 bg-indigo-50 text-indigo-700'
                  }`}
                  onClick={() => setExperimentRecoveryFilter((prev) => (prev === 'fallback_attempted' ? '' : 'fallback_attempted'))}
                  title="Filter to jobs whose latest experiment run attempted fallback verification"
                >
                  Fallback attempted {jobCountSummary.fallback_attempted}
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded-full border ${
                    experimentRecoveryFilter === 'unresolved_recovery'
                      ? 'border-rose-300 bg-rose-100 text-rose-800'
                      : 'border-rose-100 bg-rose-50 text-rose-700'
                  }`}
                  onClick={() => setExperimentRecoveryFilter((prev) => (prev === 'unresolved_recovery' ? '' : 'unresolved_recovery'))}
                  title="Filter to jobs whose latest fallback attempt remains unresolved"
                >
                  Open recovery jobs {jobCountSummary.unresolved_recovery_jobs}
                </button>
                {(jobCountSummary.pinnedOutsideFilters || jobCountSummary.pinnedOutsideList) && (
                  <button
                    type="button"
                    className="px-2 py-1 rounded-full border border-primary-100 bg-primary-50 text-primary-700 hover:bg-primary-100"
                    onClick={() => {
                      setSelectedJob(null);
                      navigate(buildAutonomousAgentsUrl(), { replace: true });
                    }}
                    title="Clear deep-linked job pin"
                  >
                    Pinned deep-linked job shown (clear)
                  </button>
                )}
              </div>

              {/* Jobs grid */}
              {jobsLoading ? (
                <div className="flex justify-center items-center flex-1">
                  <LoadingSpinner />
                </div>
              ) : (Array.isArray((jobsData as any)?.jobs) ? (jobsData as any).jobs.length : 0) === 0 ? (
                <div className="flex flex-col items-center justify-center flex-1 text-gray-500">
                  <Bot className="w-12 h-12 mb-3 text-gray-400" />
                  <p className="text-lg font-medium">No jobs yet</p>
                  <p className="text-sm">Create a new job or use a template to get started</p>
                </div>
              ) : jobsForDisplay.length === 0 ? (
                <div className="flex flex-col items-center justify-center flex-1 text-gray-500">
                  <Search className="w-12 h-12 mb-3 text-gray-400" />
                  <p className="text-lg font-medium">No jobs match current filters</p>
                  <p className="text-sm">Try clearing graph health or swarm filters</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4 overflow-y-auto flex-1">
                  {jobsForDisplay.map((job) => (
                    <JobCard
                      key={job.id}
                      job={job}
                      isPinnedDeepLink={
                        !!deepLinkedJobId &&
                        String(job.id) === String(deepLinkedJobId) &&
                        (jobCountSummary.pinnedOutsideFilters || jobCountSummary.pinnedOutsideList)
                      }
                      isSelected={selectedJob?.id === job.id}
                      onOpen={(picked) => {
                        setSelectedJob(picked);
                        navigate(buildAutonomousAgentsUrl(String(picked.id)));
                      }}
                      onOpenRunById={(jobId) => navigate(buildAutonomousAgentsUrl(jobId))}
                      onGoToQueue={() => setActiveTab('queue')}
                      onClearLaunchModeFilter={() => setLaunchModeFilter('__none__')}
                      onShowRelaunchChildren={(picked) => {
                        setHasRelaunchChildrenFilter('');
                        setRelaunchFromJobIdFilter(String(picked.id));
                      }}
                      onClearDeepLink={() =>
                        navigate(buildAutonomousAgentsUrl(), { replace: true })
                      }
                      onNarrowToSwarm={(opts) => {
                        setSwarmOnlyFilter(true);
                        if (opts?.sortBy) setSwarmSortBy(opts.sortBy);
                        if (typeof opts?.minConsensus === 'number') {
                          setSwarmMinConsensus(opts.minConsensus);
                        }
                      }}
                      onViewChainStatus={viewChainStatus}
                    />
                  ))}
                </div>
              )}
            </div>

            {/* Detail panel */}
            <div className="w-1/3">
              {selectedJob ? (
                <JobDetailPanel
                  job={selectedJob}
                  buildAutonomousAgentsUrl={buildAutonomousAgentsUrl}
                  formatDuration={formatDuration}
                  actionMutation={actionMutation}
                  createMutation={createMutation}
                  deleteMutation={deleteMutation}
                  createCodingBacklogMutation={createCodingBacklogMutation}
                  promoteDomainResearchMutation={promoteDomainResearchMutation}
                  setSelectedJob={setSelectedJob}
                  setActiveTab={setActiveTab}
                  setExportingJob={setExportingJob}
                  setShowExportModal={setShowExportModal}
                  setHasRelaunchChildrenFilter={setHasRelaunchChildrenFilter}
                  setRelaunchFromJobIdFilter={setRelaunchFromJobIdFilter}
                  swarmOutcomeByRepairJobId={swarmOutcomeByRepairJobId}
                  swarmOutcomeBySwarmJobId={swarmOutcomeBySwarmJobId}
                  unsafeExecBadge={unsafeExecBadge}
                />
              ) : (
                <div className="bg-gray-50 border border-gray-200 rounded-lg h-full flex flex-col items-center justify-center text-gray-500">
                  <Eye className="w-10 h-10 mb-3 text-gray-400" />
                  <p className="font-medium">Select a job</p>
                  <p className="text-sm">Click on a job to view details</p>
                </div>
              )}
            </div>
        </div>

        {activeTab === 'templates' && (
          <div className="w-full">
            <p className="text-sm text-gray-500 mb-4">
              Choose a template to quickly create a pre-configured autonomous job
            </p>
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  onClick={() => {
                    setTemplateRecommendScope('research');
                    if (!templateRecommendGoal.trim()) {
                      setTemplateRecommendGoal('Research a technical domain, rank evidence-backed ideas, and generate notes');
                    }
                    setShowDomainResearchQuickStartModal(true);
                  }}
                >
                  Start Domain Research
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    setTemplateRecommendScope('repo');
                    if (!templateRecommendGoal.trim()) {
                      setTemplateRecommendGoal('Run a coding swarm to reproduce the bug, rank the best repair path, and auto-launch the repair loop');
                    }
                    setShowBugTriageSwarmQuickStartModal(true);
                  }}
                >
                  Start Bug Triage Swarm
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    setTemplateRecommendScope('backend');
                    if (!templateRecommendGoal.trim()) {
                      setTemplateRecommendGoal('Diagnose the build break, isolate the failing file cluster, and auto-handoff the winning repair path');
                    }
                    setShowBuildBreakSwarmQuickStartModal(true);
                  }}
                >
                  Start Build Break Swarm
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    setTemplateRecommendScope('frontend');
                    if (!templateRecommendGoal.trim()) {
                      setTemplateRecommendGoal('Reproduce the frontend regression, isolate the affected UI surface, and promote the winning repair path');
                    }
                    setShowFrontendRegressionSwarmQuickStartModal(true);
                  }}
                >
                  Start Frontend Regression Swarm
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    setTemplateRecommendScope('repo');
                    if (!templateRecommendGoal.trim()) {
                      setTemplateRecommendGoal('Triage a repo bug from the observed symptom and return a verified patch proposal');
                    }
                    setShowRepoBugTriageQuickStartModal(true);
                  }}
                >
                  Start Repo Bug Triage
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    setTemplateRecommendScope('backend');
                    if (!templateRecommendGoal.trim()) {
                      setTemplateRecommendGoal('Fix backend API tests and stabilize integrations');
                    }
                    setShowClaudeQuickStartModal(true);
                  }}
                  disabled={!claudeBackendTemplate}
                >
                  Start Claude Backend Loop
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    if (!templateRecommendGoal.trim()) {
                      setTemplateRecommendGoal('Investigate contradictory signals and produce a validated recommendation plan');
                    }
                    setShowRoleWorkflowQuickStartModal(true);
                  }}
                >
                  Start Role Workflow
                </Button>
              </div>
              {!claudeBackendTemplate && (
                <span className="text-xs text-gray-500">Claude backend template not available</span>
              )}
            </div>
            <div className="mb-4 grid grid-cols-3 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Recommendation scope</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={templateRecommendScope}
                  onChange={(e) => setTemplateRecommendScope(e.target.value)}
                >
                  <option value="">Auto</option>
                  <option value="backend">Backend</option>
                  <option value="frontend">Frontend</option>
                  <option value="latex">LaTeX</option>
                  <option value="research">Research</option>
                </select>
              </div>
              <div className="col-span-2">
                <label className="block text-xs font-medium text-gray-600 mb-1">Goal hint (optional)</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={templateRecommendGoal}
                  onChange={(e) => setTemplateRecommendGoal(e.target.value)}
                  placeholder="e.g. Fix backend API tests for source ingestion"
                />
              </div>
            </div>
            {templatesData?.templates.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-gray-500">
                <FileText className="w-12 h-12 mb-3 text-gray-400" />
                <p className="text-lg font-medium">No templates available</p>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-4">
                {templatesData?.templates.map((template) => (
                  <TemplateCard
                    key={template.id}
                    template={template}
                    typeConfig={
                      JOB_TYPE_CONFIG[template.job_type as AgentJobType] ||
                      JOB_TYPE_CONFIG.custom
                    }
                    onSelect={setCreateFromTemplate}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'chains' && (
          <div className="w-full">
            <p className="text-sm text-gray-500 mb-4">
              Job chains allow you to create multi-step workflows where jobs automatically trigger subsequent jobs on completion
            </p>
            {displayedChainDefinitions.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-gray-500">
                <GitBranch className="w-12 h-12 mb-3 text-gray-400" />
                <p className="text-lg font-medium">No chain definitions yet</p>
                <p className="text-sm">Chain definitions allow you to create multi-step workflows</p>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-4">
                {displayedChainDefinitions.map((chain) => {
                  const isRecoveryPlaybook = String(chain.name || '').toLowerCase().startsWith('playbook_recovery_')
                    || String(chain.display_name || '').toLowerCase().includes('recovery playbook')
                    || String(chain.description || '').toLowerCase().includes('saved as a recovery playbook');
                  return (
                  <div
                    key={chain.id}
                    className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-start gap-3 mb-3">
                      <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
                        <GitBranch className="w-5 h-5" />
                      </div>
                      <div className="flex-1">
                        <h3 className="section-heading">{chain.display_name}</h3>
                        <p className="text-sm text-gray-500">{chain.chain_steps.length} steps</p>
                      </div>
                      {isRecoveryPlaybook ? (
                        <span className="text-xs bg-amber-100 text-amber-800 px-2 py-1 rounded">Recovery</span>
                      ) : null}
                      {chain.is_system && (
                        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">System</span>
                      )}
                    </div>
                    {chain.description && (
                      <p className="text-sm text-gray-600 mb-3 line-clamp-2">{chain.description}</p>
                    )}
                    <div className="flex flex-wrap gap-2 mb-3">
                      {chain.chain_steps.slice(0, 3).map((step, idx) => (
                        <span key={idx} className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                          {step.step_name}
                        </span>
                      ))}
                      {chain.chain_steps.length > 3 && (
                        <span className="text-xs text-gray-500">+{chain.chain_steps.length - 3} more</span>
                      )}
                    </div>
                    <Button
                      size="sm"
                      variant="secondary"
                      className="w-full"
                      onClick={() => {
                        setStartFromChain(chain);
                      }}
                    >
                      <Play className="w-3 h-3 mr-1" />
                      Start Chain
                    </Button>
                  </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {activeTab === 'inbox' && (
          <div className="w-full flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3 text-sm text-gray-600">
                <span className="font-medium text-gray-900">Research Inbox</span>
                <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded">Total: {inboxStats?.total ?? '-'}</span>
                <span className="bg-primary-100 text-primary-700 px-2 py-1 rounded">New: {inboxStats?.new ?? '-'}</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded">Accepted: {inboxStats?.accepted ?? '-'}</span>
                <span className="bg-red-100 text-red-700 px-2 py-1 rounded">Rejected: {inboxStats?.rejected ?? '-'}</span>
              </div>
              <div className="flex gap-2">
                <label className="flex items-center gap-2 text-xs text-gray-600 select-none" title={paperAlgoDefaultToggleTitle}>
                  <input
                    type="checkbox"
                    className="h-3 w-3"
                    checked={paperAlgoDefaultRunDemoCheck}
                    disabled={updateMyPreferencesMutation.isLoading}
                    onChange={(e) => updateMyPreferencesMutation.mutate({ paper_algo_default_run_demo_check: e.target.checked })}
                  />
                  <span>Default: Run demo check</span>
                </label>
                <Button variant="secondary" onClick={() => setShowInboxMonitorModal(true)}>
                  <Activity className="w-4 h-4 mr-2" />
                  Create Monitor
                </Button>
                <Button variant="secondary" onClick={() => setShowMonitorProfilesModal(true)}>
                  <Settings className="w-4 h-4 mr-2" />
                  Profiles
                </Button>
                <Button variant="ghost" onClick={() => refetchInbox()}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <div className="flex gap-3 mb-4">
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={inboxStatusFilter}
                onChange={(e) => setInboxStatusFilter(e.target.value as any)}
              >
                <option value="">All Status</option>
                <option value="new">New</option>
                <option value="accepted">Accepted</option>
                <option value="rejected">Rejected</option>
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={inboxTypeFilter}
                onChange={(e) => setInboxTypeFilter(e.target.value)}
              >
                <option value="">All Types</option>
                <option value="document">Document</option>
                <option value="arxiv">arXiv</option>
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={inboxCustomerFilter}
                onChange={(e) => setInboxCustomerFilter(e.target.value)}
              >
                <option value="">All Customers</option>
                {healthCustomers.map((customer) => (
                  <option key={customer} value={customer}>
                    {customer}
                  </option>
                ))}
              </select>
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    className="w-full border border-gray-300 rounded-lg pl-9 pr-3 py-2 text-sm"
                    placeholder="Search inbox items…"
                    value={inboxSearch}
                    onChange={(e) => setInboxSearch(e.target.value)}
                  />
              </div>
            </div>

            {inboxCustomerFilter ? (
              <div className="flex items-center gap-2 mb-4 text-xs">
                <span className="bg-amber-50 text-amber-800 border border-amber-200 px-2 py-1 rounded">
                  Customer filter: {inboxCustomerFilter}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setInboxCustomerFilter('');
                    navigate(buildAutonomousAgentsUrl(undefined, {
                      inbox_customer: null,
                    }), { replace: true });
                  }}
                >
                  Clear customer filter
                </Button>
              </div>
            ) : null}

            {inboxJobFilter ? (
              <div className="flex items-center gap-2 mb-4 text-xs">
                <span className="bg-sky-50 text-sky-800 border border-sky-200 px-2 py-1 rounded">
                  Monitor filter: {inboxJobFilter}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setInboxJobFilter('');
                    const params = new URLSearchParams(location.search);
                    params.delete('inbox_job');
                    params.delete('inbox');
                    navigate(`${location.pathname}${params.toString() ? `?${params.toString()}` : ''}`, { replace: true });
                  }}
                >
                  Clear monitor filter
                </Button>
              </div>
            ) : null}

            {inboxHealthDrilldown ? (
              <div className="flex items-center gap-2 mb-4 text-xs">
                <span className="bg-emerald-50 text-emerald-800 border border-emerald-200 px-2 py-1 rounded">
                  Showing accepted follow-ups
                  {inboxCustomerFilter ? ` for ${inboxCustomerFilter}` : ''}
                  {inboxJobFilter ? ` · ${inboxJobFilter}` : ''}
                  {` · ${formatInboxHealthDrilldownLabel(inboxHealthDrilldown)}`}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setInboxHealthDrilldown('');
                    navigate(buildAutonomousAgentsUrl(undefined, {
                      inbox_health_drilldown: null,
                    }), { replace: true });
                  }}
                >
                  Clear drilldown
                </Button>
              </div>
            ) : null}

            {inboxPolicyDrilldown ? (
              <div className="flex items-center gap-2 mb-4 text-xs">
                <span className="bg-violet-50 text-violet-800 border border-violet-200 px-2 py-1 rounded">
                  Showing accepted signals
                  {inboxJobFilter ? ` for ${inboxJobFilter}` : ''}
                  {` · ${formatInboxPolicyDrilldownLabel(inboxPolicyDrilldown)}`}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setInboxPolicyDrilldown('');
                    navigate(buildAutonomousAgentsUrl(undefined, {
                      inbox_policy_drilldown: null,
                    }), { replace: true });
                  }}
                >
                  Clear drilldown
                </Button>
              </div>
            ) : null}

            {(() => {
              const items = visibleInboxItems;
              const selectedIds = selectedInboxItems.map((item) => String(item.id));
              const allSelected = items.length > 0 && selectedIds.length === items.length;
              if (items.length === 0) return null;
              return (
                <div className="flex items-center justify-between mb-3 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={allSelected}
                      onChange={(e) => {
                        const next: Record<string, boolean> = {};
                        if (e.target.checked) {
                          items.forEach((it) => (next[it.id] = true));
                        }
                        setSelectedInboxIds(next);
                      }}
                    />
                    <span className="text-sm text-gray-700">
                      Selected: {selectedIds.length}/{items.length}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={selectedIds.length === 0 || createMutation.isLoading || createFromChainMutation.isLoading}
                    onClick={() => {
                        const selectedItems = selectedInboxItems;
                        if (selectedItems.length === 0) return;

                        const goal = inboxResearchGoalDraft.trim();
                        if (!goal.trim()) return;

                        const docItems = selectedItems.filter((it) => it.item_type === 'document');
                        const paperItems = selectedItems.filter((it) => it.item_type === 'arxiv');

                        const top_documents = docItems.slice(0, 20).map((d) => ({
                          id: d.item_key,
                          title: d.title,
                          url: d.url,
                          score: null,
                          source: 'inbox',
                        }));
                        const top_papers = paperItems.slice(0, 20).map((p) => ({
                          id: p.item_key,
                          title: p.title,
                          url: p.url,
                          score: null,
                          source: 'inbox',
                        }));

                        const parent_findings = selectedItems.slice(0, 50).map((it) => ({
                          type: it.item_type === 'arxiv' ? 'paper' : 'document',
                          title: it.title,
                          id: it.item_key,
                          url: it.url,
                          snippet: it.summary,
                        }));

                        const customers = Array.from(new Set(selectedItems.map((it) => it.customer).filter(Boolean))) as string[];
                        const customerContextHint =
                          customers.length === 1 ? `Customer: ${customers[0]}` : customers.length > 1 ? `Customers: ${customers.join(', ')}` : '';

                        const chains = ((chainsData as any)?.chains || []) as any[];
                        const deepDiveChain =
                          chains.find((c: any) => c?.name === 'customer_research_scout_deep_dive_chain') || null;

                        if (deepDiveChain?.id) {
                          createFromChainMutation.mutate({
                            chain_definition_id: deepDiveChain.id,
                            name_prefix: `Inbox Research — ${new Date().toLocaleDateString()}`,
                            variables: { goal: goal.trim() },
                            config_overrides: {
                              customer_context: customerContextHint,
                              prefer_sources: ['documents', 'arxiv'],
                              max_documents: 12,
                              max_papers: 8,
                              persist_artifacts: false,
                              reading_list_name: 'Customer Research',
                              inherited_data: {
                                parent_results: {
                                  summary: `Seeded from ${selectedItems.length} Research Inbox items.`,
                                  research_bundle: {
                                    top_documents,
                                    top_papers,
                                    insights: [],
                                    next_steps: [],
                                    artifacts: [],
                                  },
                                  inbox_items: selectedItems,
                                },
                                parent_findings,
                              },
                            },
                            start_immediately: true,
                          });
                        } else {
                          // Fallback: single research job
                          createMutation.mutate({
                            name: `Inbox Research — ${new Date().toLocaleDateString()}`,
                            job_type: 'research',
                            goal: goal.trim(),
                            config: {
                              customer_context: customerContextHint,
                              prefer_sources: ['documents', 'arxiv'],
                              max_documents: 12,
                              max_papers: 8,
                              persist_artifacts: false,
                              reading_list_name: 'Customer Research',
                              inherited_data: {
                                parent_results: {
                                  summary: `Seeded from ${selectedItems.length} Research Inbox items.`,
                                  research_bundle: {
                                    top_documents,
                                    top_papers,
                                    insights: [],
                                    next_steps: [],
                                    artifacts: [],
                                  },
                                  inbox_items: selectedItems,
                                },
                                parent_findings,
                              },
                            },
                            start_immediately: true,
                          });
                        }
                        setActiveTab('jobs');
                      }}
                    >
                      <Sparkles className="w-4 h-4 mr-1" />
                      Research Selected
                    </Button>
                    <Button
                      size="sm"
                      variant="primary"
                      disabled={!inboxBulkFollowUpState.enabled || bulkInboxFollowUpActionMutation.isLoading}
                      onClick={() => {
                        if (!inboxBulkFollowUpState.enabled) return;
                        bulkInboxFollowUpActionMutation.mutate({
                          domain_research_profile_id: inboxBulkFollowUpState.ownerKind === 'domain' ? inboxBulkFollowUpState.ownerId : undefined,
                          profile_opportunity_ids: inboxBulkFollowUpState.ownerKind === 'domain' ? inboxBulkFollowUpState.opportunityIds : undefined,
                          portfolio_id: inboxBulkFollowUpState.ownerKind === 'fleet' ? inboxBulkFollowUpState.ownerId : undefined,
                          portfolio_opportunity_ids: inboxBulkFollowUpState.ownerKind === 'fleet' ? inboxBulkFollowUpState.opportunityIds : undefined,
                          action: 'approve_launch',
                          operator_note: inboxBulkFollowUpNote.trim() || undefined,
                        });
                      }}
                    >
                      <ThumbsUp className="w-4 h-4 mr-1" />
                      Approve Follow-ups
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={!inboxBulkFollowUpState.enabled || bulkInboxFollowUpActionMutation.isLoading}
                      onClick={() => {
                        if (!inboxBulkFollowUpState.enabled) return;
                        bulkInboxFollowUpActionMutation.mutate({
                          domain_research_profile_id: inboxBulkFollowUpState.ownerKind === 'domain' ? inboxBulkFollowUpState.ownerId : undefined,
                          profile_opportunity_ids: inboxBulkFollowUpState.ownerKind === 'domain' ? inboxBulkFollowUpState.opportunityIds : undefined,
                          portfolio_id: inboxBulkFollowUpState.ownerKind === 'fleet' ? inboxBulkFollowUpState.ownerId : undefined,
                          portfolio_opportunity_ids: inboxBulkFollowUpState.ownerKind === 'fleet' ? inboxBulkFollowUpState.opportunityIds : undefined,
                          action: 'reject_launch',
                          operator_note: inboxBulkFollowUpNote.trim() || undefined,
                        });
                      }}
                    >
                      <ThumbsDown className="w-4 h-4 mr-1" />
                      Reject Follow-ups
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={!inboxBulkRelaunchState.enabled || bulkInboxRelaunchMutation.isLoading}
                      onClick={() => {
                        if (!inboxBulkRelaunchState.enabled) return;
                        bulkInboxRelaunchMutation.mutate({
                          item_ids: inboxBulkRelaunchState.itemIds,
                          operator_note: inboxBulkFollowUpNote.trim() || undefined,
                        });
                      }}
                    >
                      <RotateCcw className="w-4 h-4 mr-1" />
                      Relaunch Follow-ups
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={selectedIds.length === 0 || bulkUpdateInboxMutation.isLoading}
                      onClick={() => bulkUpdateInboxMutation.mutate({ itemIds: selectedIds, data: { status: 'accepted' } })}
                    >
                      <ThumbsUp className="w-4 h-4 mr-1" />
                      Accept Selected
                    </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={selectedIds.length === 0 || bulkUpdateInboxMutation.isLoading}
                    onClick={() => {
                        bulkUpdateInboxMutation.mutate({
                          itemIds: selectedIds,
                          data: { status: 'rejected', feedback: inboxBulkRejectReason.trim() || undefined },
                        });
                        setInboxBulkRejectReason('');
                      }}
                  >
                    <ThumbsDown className="w-4 h-4 mr-1" />
                    Reject Selected
                  </Button>
                </div>
                <textarea
                  className="mt-3 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={3}
                  placeholder="Follow-up research goal"
                  value={inboxResearchGoalDraft}
                  onChange={(e) => setInboxResearchGoalDraft(e.target.value)}
                />
                <div className="mt-3 grid gap-2 md:grid-cols-[minmax(0,1fr)_auto]">
                  <input
                    className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    placeholder="Bulk follow-up note (optional)"
                    value={inboxBulkFollowUpNote}
                    onChange={(e) => setInboxBulkFollowUpNote(e.target.value)}
                  />
                  <div className="text-[11px] text-gray-500 self-center">
                    {inboxBulkFollowUpState.enabled
                      ? 'Applies to selected pending follow-up approvals'
                      : inboxBulkRelaunchState.enabled
                        ? 'Applies to selected failed or cancelled follow-ups'
                        : inboxBulkFollowUpState.disabledReason || inboxBulkRelaunchState.disabledReason}
                  </div>
                </div>
                <div className="mt-3 grid gap-2 md:grid-cols-[minmax(0,1fr)_auto]">
                  <input
                    className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    placeholder="Bulk reject reason (optional)"
                    value={inboxBulkRejectReason}
                    onChange={(e) => setInboxBulkRejectReason(e.target.value)}
                  />
                  <div className="text-[11px] text-gray-500 self-center">
                    Applies when rejecting selected inbox items
                  </div>
                </div>
              </div>
              );
            })()}

            {inboxLoading ? (
              <div className="flex justify-center items-center flex-1">
                <LoadingSpinner />
              </div>
            ) : visibleInboxItems.length === 0 ? (
              <div className="flex flex-col items-center justify-center flex-1 text-gray-500">
                <Inbox className="w-12 h-12 mb-3 text-gray-400" />
                <p className="text-lg font-medium">Inbox is empty</p>
                <p className="text-sm">Create a monitor or run customer research to discover items</p>
              </div>
            ) : (
              <div className="space-y-3 overflow-y-auto flex-1 pr-1">
                {visibleInboxItems.map((item: ResearchInboxItem) => (
                  <div
                    key={item.id}
                    className={`bg-white border rounded-lg p-4 ${
                      deepLinkedInboxId && String(deepLinkedInboxId) === String(item.id)
                        ? 'border-emerald-400 ring-1 ring-emerald-200'
                        : 'border-gray-200'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={!!selectedInboxIds[item.id]}
                            onChange={(e) => setSelectedInboxIds((prev) => ({ ...prev, [item.id]: e.target.checked }))}
                          />
                          <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
                            {item.item_type}
                          </span>
                          <span
                            className={`text-xs px-2 py-1 rounded ${
                              item.status === 'new'
                                ? 'bg-primary-100 text-primary-700'
                                : item.status === 'accepted'
                                ? 'bg-green-100 text-green-700'
                                : 'bg-red-100 text-red-700'
                            }`}
                          >
                            {item.status}
                          </span>
                          {item.customer ? (
                            <span className="text-xs bg-gray-50 text-gray-600 px-2 py-1 rounded">
                              {item.customer}
                            </span>
                          ) : null}
                          {item.follow_up_launch_status ? (
                            <span
                              className={`text-xs px-2 py-1 rounded ${
                                item.follow_up_launch_status === 'launched'
                                  ? 'bg-emerald-100 text-emerald-700'
                                  : item.follow_up_launch_status === 'pending_approval'
                                    ? 'bg-amber-100 text-amber-800'
                                    : item.follow_up_launch_status === 'failed'
                                      ? 'bg-rose-100 text-rose-700'
                                      : 'bg-slate-100 text-slate-700'
                              }`}
                            >
                              {item.follow_up_launch_status.replace(/_/g, ' ')}
                            </span>
                          ) : null}
                          {item.follow_up_outcome_status ? (
                            <span
                              className={`text-xs px-2 py-1 rounded ${
                                item.follow_up_outcome_status === 'completed'
                                  ? 'bg-blue-100 text-blue-700'
                                  : item.follow_up_outcome_status === 'failed'
                                    ? 'bg-rose-100 text-rose-700'
                                    : 'bg-slate-100 text-slate-700'
                              }`}
                            >
                              outcome: {item.follow_up_outcome_status.replace(/_/g, ' ')}
                            </span>
                          ) : null}
                        </div>
                        <h3 className="section-heading mt-2 truncate">{item.title}</h3>
                        {item.summary ? (
                          <p className="text-sm text-gray-600 mt-1 line-clamp-2">{item.summary}</p>
                        ) : null}
                        {(item.metadata as any)?.query ? (
                          <p className="text-xs text-gray-500 mt-2">
                            Why: matched query “{String((item.metadata as any).query).slice(0, 140)}”
                            {(item.metadata as any)?.bias?.source ? (
                              <span className="ml-2 bg-gray-100 text-gray-700 px-2 py-0.5 rounded">
                                bias: {String((item.metadata as any).bias.source)}
                              </span>
                            ) : null}
                          </p>
                        ) : null}
                        <div className="text-xs text-gray-500 mt-2 flex flex-wrap gap-x-4 gap-y-1">
                          <span>Discovered: {new Date(item.discovered_at).toLocaleString()}</span>
                          {item.published_at ? (
                            <span>Published: {new Date(item.published_at).toLocaleDateString()}</span>
                          ) : null}
                          {item.follow_up_policy_mode ? (
                            <span>Policy: {item.follow_up_policy_mode.replace(/_/g, ' ')}</span>
                          ) : null}
                          {typeof (item.metadata as any)?.discovery_score === 'number' ? (
                            <span>Discovery score: {Number((item.metadata as any).discovery_score)}</span>
                          ) : null}
                          {item.follow_up_launched_at ? (
                            <span>Launched: {new Date(item.follow_up_launched_at).toLocaleString()}</span>
                          ) : null}
                          {item.follow_up_outcome_recorded_at ? (
                            <span>Outcome: {new Date(item.follow_up_outcome_recorded_at).toLocaleString()}</span>
                          ) : null}
                          {item.follow_up_operator_acted_at ? (
                            <span>Operator acted: {new Date(item.follow_up_operator_acted_at).toLocaleString()}</span>
                          ) : null}
                        </div>
                        {item.follow_up_block_reason ? (
                          <p className="text-xs text-gray-500 mt-2">
                            Follow-up: {item.follow_up_block_reason}
                          </p>
                        ) : null}
                        {item.follow_up_budget_decision || item.follow_up_budget_throttle_state ? (
                          <p className="text-xs text-amber-700 mt-2">
                            Budget: {String(item.follow_up_budget_decision || item.follow_up_budget_throttle_state || '').replace(/_/g, ' ')}
                            {item.follow_up_budget_reason ? ` — ${item.follow_up_budget_reason}` : ''}
                          </p>
                        ) : null}
                        {item.follow_up_customer_budget_decision || item.follow_up_customer_budget_throttle_state ? (
                          <p className="text-xs text-rose-700 mt-2">
                            Customer budget: {String(item.follow_up_customer_budget_decision || item.follow_up_customer_budget_throttle_state || '').replace(/_/g, ' ')}
                            {item.follow_up_customer_budget_reason ? ` — ${item.follow_up_customer_budget_reason}` : ''}
                          </p>
                        ) : null}
                        {Array.isArray((item.metadata as any)?.discovery_reasons) && (item.metadata as any).discovery_reasons.length > 0 ? (
                          <p className="text-xs text-gray-500 mt-2">
                            Discovery why: {((item.metadata as any).discovery_reasons as string[]).slice(0, 3).join(', ')}
                          </p>
                        ) : null}
                        {item.follow_up_operator_decision ? (
                          <p className="text-xs text-gray-500 mt-2">
                            Operator: {item.follow_up_operator_decision.replace(/_/g, ' ')}
                            {item.follow_up_operator_note ? ` — ${item.follow_up_operator_note}` : ''}
                          </p>
                        ) : null}
                        {item.follow_up_outcome_summary ? (
                          <p className="text-xs text-gray-500 mt-2">
                            Outcome summary: {item.follow_up_outcome_summary}
                          </p>
                        ) : null}
                        {item.status === 'accepted'
                        && item.item_type === 'follow_up_recommendation'
                        && String(item.follow_up_launch_status || '').trim().toLowerCase() === 'pending_approval' ? (
                          <input
                            aria-label={`Inbox follow-up note for ${String(item.title || item.id)}`}
                            className="mt-3 border border-gray-300 rounded-lg px-3 py-2 text-sm w-full"
                            placeholder="Follow-up note (optional)"
                            value={followUpReviewNoteDrafts[`inbox:${String(item.id)}`] || ''}
                            disabled={followUpQueueActionMutation.isLoading && activeFollowUpReviewKey === `inbox:${String(item.id)}`}
                            onChange={(e) => setFollowUpReviewNoteDrafts((prev) => ({ ...prev, [`inbox:${String(item.id)}`]: e.target.value }))}
                          />
                        ) : null}
                        {item.origin_source_kind && item.origin_source_id && item.origin_opportunity_id ? (
                          <p className="text-xs text-gray-500 mt-2">
                            Target: {String(item.origin_source_kind).trim().toLowerCase() === 'profile' ? 'Domain profile' : 'Research fleet'}
                          </p>
                        ) : null}
                      </div>
                      <div className="flex flex-col gap-2 shrink-0">
                        {item.origin_source_kind && item.origin_source_id && item.origin_opportunity_id ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => {
                              const sourceKind = String(item.origin_source_kind || '').trim().toLowerCase();
                              const extras = sourceKind === 'profile'
                                ? {
                                    tab: 'domain',
                                    profileId: String(item.origin_source_id || ''),
                                    opportunityId: String(item.origin_opportunity_id || ''),
                                  }
                                : {
                                    tab: 'fleet',
                                    fleetId: String(item.origin_source_id || ''),
                                    opportunityId: String(item.origin_opportunity_id || ''),
                                  };
                              setActiveTab(sourceKind === 'profile' ? 'domain' : 'fleet');
                              navigate(buildAutonomousAgentsUrl(undefined, extras), { replace: true });
                            }}
                          >
                            Open Target
                          </Button>
                        ) : null}
                        {item.follow_up_job_id ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => {
                              setActiveTab('jobs');
                              navigate(buildAutonomousAgentsUrl(item.follow_up_job_id), { replace: true });
                            }}
                          >
                            Open Follow-up
                          </Button>
                        ) : null}
                        {item.follow_up_launch_status === 'launched' && ['failed', 'cancelled'].includes(String(item.follow_up_outcome_status || '').trim().toLowerCase()) ? (
                          <Button
                            size="sm"
                            variant="secondary"
                            disabled={relaunchInboxFollowUpMutation.isLoading}
                            onClick={() => relaunchInboxFollowUpMutation.mutate({ itemId: item.id })}
                          >
                            <RotateCcw className="w-4 h-4 mr-1" />
                            Relaunch Follow-up
                          </Button>
                        ) : null}
                        {item.status === 'accepted'
                        && item.item_type === 'follow_up_recommendation'
                        && String(item.follow_up_launch_status || '').trim().toLowerCase() === 'pending_approval' ? (
                          <>
                            <Button
                              size="sm"
                              variant="primary"
                              disabled={followUpQueueActionMutation.isLoading && activeFollowUpReviewKey === `inbox:${String(item.id)}`}
                              onClick={() => followUpQueueActionMutation.mutate({
                                inbox_item_id: String(item.id),
                                action: 'approve_launch',
                                operator_note: followUpReviewNoteDrafts[`inbox:${String(item.id)}`]?.trim() || undefined,
                                navigateOnLaunch: false,
                                reviewRowKey: `inbox:${String(item.id)}`,
                              })}
                            >
                              <ThumbsUp className="w-4 h-4 mr-1" />
                              Approve Follow-up
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={followUpQueueActionMutation.isLoading && activeFollowUpReviewKey === `inbox:${String(item.id)}`}
                              onClick={() => followUpQueueActionMutation.mutate({
                                inbox_item_id: String(item.id),
                                action: 'reject_launch',
                                operator_note: followUpReviewNoteDrafts[`inbox:${String(item.id)}`]?.trim() || undefined,
                                navigateOnLaunch: false,
                                reviewRowKey: `inbox:${String(item.id)}`,
                              })}
                            >
                              <ThumbsDown className="w-4 h-4 mr-1" />
                              Reject Follow-up
                            </Button>
                          </>
                        ) : null}
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={item.status === 'accepted' || updateInboxItemMutation.isLoading}
                          onClick={() => updateInboxItemMutation.mutate({ itemId: item.id, data: { status: 'accepted' } })}
                        >
                          <ThumbsUp className="w-4 h-4 mr-1" />
                          Accept
                        </Button>
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={item.status === 'rejected' || updateInboxItemMutation.isLoading}
                          onClick={() => {
                            updateInboxItemMutation.mutate({
                              itemId: item.id,
                              data: { status: 'rejected', feedback: inboxRejectReasonDrafts[item.id]?.trim() || undefined },
                            });
                          }}
                        >
                          <ThumbsDown className="w-4 h-4 mr-1" />
                          Reject
                        </Button>
                        <div className="grid gap-2">
                          <input
                            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                            placeholder="Reject reason (optional)"
                            value={inboxRejectReasonDrafts[item.id] || ''}
                            onChange={(e) => setInboxRejectReasonDrafts((current) => ({ ...current, [item.id]: e.target.value }))}
                          />
                          <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_auto]">
                            <input
                              className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                              placeholder="Mute token"
                              value={inboxMuteTokenDrafts[item.id] ?? ((item.title || '').split(/[^a-zA-Z0-9_-]+/).find((t) => t && t.length >= 4) || '')}
                              onChange={(e) => setInboxMuteTokenDrafts((current) => ({ ...current, [item.id]: e.target.value }))}
                            />
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={upsertMonitorProfileMutation.isLoading}
                              onClick={() => {
                                const token = String(inboxMuteTokenDrafts[item.id] || '').trim().toLowerCase();
                                if (!token) {
                                  toast.error('Enter a mute token first');
                                  return;
                                }
                                upsertMonitorProfileMutation.mutate({ customer: item.customer || undefined, muted_tokens: [token], merge_lists: true });
                              }}
                            >
                              Mute token
                            </Button>
                          </div>
                          <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_auto]">
                            <input
                              className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                              placeholder="Mute phrase"
                              value={inboxMutePhraseDrafts[item.id] ?? (item.title || '')}
                              onChange={(e) => setInboxMutePhraseDrafts((current) => ({ ...current, [item.id]: e.target.value }))}
                            />
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={upsertMonitorProfileMutation.isLoading}
                              onClick={() => {
                                const phrase = String(inboxMutePhraseDrafts[item.id] || '').trim();
                                if (!phrase) {
                                  toast.error('Enter a mute phrase first');
                                  return;
                                }
                                upsertMonitorProfileMutation.mutate({ customer: item.customer || undefined, muted_patterns: [phrase], merge_lists: true });
                              }}
                            >
                              Mute phrase
                            </Button>
                          </div>
                        </div>
                        {(item.metadata as any)?.query ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={upsertMonitorProfileMutation.isLoading}
                            onClick={() => {
                              const q = String((item.metadata as any).query || '').trim();
                              if (!q) return;
                              upsertMonitorProfileMutation.mutate({ customer: item.customer || undefined, muted_patterns: [q], merge_lists: true });
                            }}
                          >
                            Mute query
                          </Button>
                        ) : null}
                            {item.item_type === 'arxiv' ? (
                              <>
                            {Array.isArray((item.metadata as any)?.repos) && (item.metadata as any).repos.length > 0 ? (
                              <>
                                <div className="text-xs text-gray-500 mt-1">Repos</div>
                                {((item.metadata as any).repos as any[]).slice(0, 2).map((r: any, idx: number) => (
                                  <Button
                                    key={idx}
                                    size="sm"
                                    variant="secondary"
                                    disabled={ingestRepoMutation.isLoading || String(r?.provider) !== 'github'}
                                    title={String(r?.provider) === 'github' ? 'Ingest this repo' : 'GitLab ingestion requires a token (use Documents → Repos)'}
                                    onClick={() => ingestRepoMutation.mutate({ provider: 'github', repo: String(r.repo) })}
                                  >
                                    Ingest {String(r?.provider || 'repo')}
                                  </Button>
                                ))}
                              </>
                            ) : (
                              <Button
                                size="sm"
                                variant="secondary"
                                disabled={extractReposMutation.isLoading}
                                onClick={() => extractReposMutation.mutate(item.id)}
                              >
                                Find repos
                              </Button>
                            )}
                            {Array.isArray((item.metadata as any)?.repos) && (item.metadata as any).repos.filter((r: any) => String(r?.provider || '').toLowerCase() === 'github').length > 1 ? (
                              <select
                                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                                value={
                                  String(
                                    paperRepoSelectionDrafts[item.id] ??
                                      String(((item.metadata as any).repos as any[]).find((r: any) => String(r?.provider || '').toLowerCase() === 'github')?.repo || '')
                                  )
                                }
                                onChange={(e) => setPaperRepoSelectionDrafts((current) => ({ ...current, [item.id]: e.target.value }))}
                              >
                                {((item.metadata as any).repos as any[])
                                  .filter((r: any) => String(r?.provider || '').toLowerCase() === 'github')
                                  .slice(0, 12)
                                  .map((r: any) => (
                                    <option key={String(r.repo)} value={String(r.repo)}>
                                      {String(r.repo)}
                                    </option>
                                  ))}
                              </select>
                            ) : null}
                            <Button
                              size="sm"
                              variant="secondary"
                              disabled={createFromChainMutation.isLoading}
                              onClick={() =>
                                runPaperRepoCodeAgent(
                                  item,
                                  paperRepoSelectionDrafts[item.id] ||
                                    String(
                                      Array.isArray((item.metadata as any)?.repos)
                                        ? ((item.metadata as any).repos as any[]).find((r: any) => String(r?.provider || '').toLowerCase() === 'github')?.repo || ''
                                        : ''
                                    )
                                )
                              }
                            >
                              Code Agent on Repo
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              disabled={createFromChainMutation.isLoading}
                              onClick={() => {
                                const persistedEp = String((item.metadata as any)?.paper_algo_entrypoint || '').trim();
                                const ep = (paperAlgoEntrypoint[item.id] ?? persistedEp ?? 'demo.py') || 'demo.py';
                                runPaperAlgorithmProject(item, paperAlgoRunDemo[item.id] ?? paperAlgoDefaultRunDemoCheck, ep);
                              }}
                            >
                              <span className="inline-flex items-center gap-2">
                                <span>Implement Algorithm</span>
                                <span className="inline-flex items-center gap-1" title={unsafeExecBadge.title}>
                                  <span className={`inline-block w-2 h-2 rounded-full ${unsafeExecBadge.color}`} />
                                  <span className="text-[10px] text-gray-600">{unsafeExecBadge.label}</span>
                                </span>
                              </span>
                            </Button>
                            <label
                              className="flex items-center gap-1 text-xs text-gray-600 select-none"
                              title={
                                unsafeExecBadge.status === 'ready'
                                  ? 'Run a sandboxed demo.py check after generating the project'
                                  : unsafeExecBadge.title
                              }
                            >
                              {(() => {
                                const persisted = (item.metadata as any)?.paper_algo_run_demo_check;
                                const checked =
                                  typeof persisted === 'boolean'
                                    ? persisted
                                    : paperAlgoRunDemo[item.id] ?? paperAlgoDefaultRunDemoCheck;
                                return (
                              <input
                                type="checkbox"
                                className="h-3 w-3"
                                checked={checked}
                                disabled={unsafeExecBadge.status !== 'ready' || updateInboxItemMutation.isLoading}
                                onChange={(e) => {
                                  const v = e.target.checked;
                                  setPaperAlgoRunDemo((prev) => ({ ...prev, [item.id]: v }));
                                  updateInboxItemMutation.mutate({
                                    itemId: item.id,
                                    data: { metadata_patch: { paper_algo_run_demo_check: v } },
                                  });
                                }}
                              />
                                );
                              })()}
                              <span>Run demo check</span>
                            </label>
                            <input
                              className={`border rounded px-2 py-1 text-xs w-36 ${
                                paperAlgoEntrypointError[item.id] ? 'border-red-400' : 'border-gray-200'
                              }`}
                              placeholder="demo.py"
                              value={
                                paperAlgoEntrypoint[item.id] ??
                                (String((item.metadata as any)?.paper_algo_entrypoint || '').trim() || 'demo.py')
                              }
                              onChange={(e) => {
                                const raw = e.target.value;
                                setPaperAlgoEntrypoint((prev) => ({ ...prev, [item.id]: raw }));
                                const check = normalizeEntrypoint(raw);
                                setPaperAlgoEntrypointError((prev) => ({ ...prev, [item.id]: check.ok ? '' : String(check.error || 'Invalid') }));
                              }}
                              onBlur={async () => {
                                const raw =
                                  paperAlgoEntrypoint[item.id] ??
                                  (String((item.metadata as any)?.paper_algo_entrypoint || '').trim() || 'demo.py');
                                const check = normalizeEntrypoint(raw);
                                setPaperAlgoEntrypoint((prev) => ({ ...prev, [item.id]: check.value }));
                                setPaperAlgoEntrypointError((prev) => ({ ...prev, [item.id]: check.ok ? '' : String(check.error || 'Invalid') }));
                                if (!check.ok) {
                                  toast.error(`Invalid entrypoint: ${check.error || 'Invalid'}`);
                                  return;
                                }
                                setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: true }));
                                try {
                                  await apiClient.updateResearchInboxItem(item.id, {
                                    metadata_patch: { paper_algo_entrypoint: check.value },
                                  } as any);
                                  queryClient.invalidateQueries(['research-inbox']);
                                  setPaperAlgoEntrypointSavedAt((prev) => ({ ...prev, [item.id]: new Date().toISOString() }));
                                } catch (e: any) {
                                  toast.error(e?.response?.data?.detail || e?.message || 'Failed to save entrypoint');
                                } finally {
                                  setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: false }));
                                }
                              }}
                              title={
                                paperAlgoEntrypointError[item.id]
                                  ? `Entrypoint invalid: ${paperAlgoEntrypointError[item.id]}`
                                  : 'Demo entrypoint path (persisted per paper)'
                              }
                            />
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={paperAlgoEntrypointSaving[item.id] || updateInboxItemMutation.isLoading}
                              title="Reset entrypoint override to default (demo.py)"
                              onClick={async () => {
                                setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: true }));
                                try {
                                  await apiClient.updateResearchInboxItem(item.id, {
                                    metadata_patch: { paper_algo_entrypoint: null },
                                  } as any);
                                  setPaperAlgoEntrypoint((prev) => {
                                    const next = { ...prev };
                                    delete next[item.id];
                                    return next;
                                  });
                                  setPaperAlgoEntrypointError((prev) => ({ ...prev, [item.id]: '' }));
                                  setPaperAlgoEntrypointSavedAt((prev) => ({ ...prev, [item.id]: new Date().toISOString() }));
                                  queryClient.invalidateQueries(['research-inbox']);
                                  toast.success('Entrypoint reset to default');
                                } catch (e: any) {
                                  toast.error(e?.response?.data?.detail || e?.message || 'Failed to reset entrypoint');
                                } finally {
                                  setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: false }));
                                }
                              }}
                            >
                              Reset
                            </Button>
                            <span className="text-[10px] text-gray-500 min-w-[42px]">
                              {paperAlgoEntrypointSaving[item.id]
                                ? 'saving…'
                                : paperAlgoEntrypointSavedAt[item.id]
                                  ? 'saved'
                                  : ''}
                            </span>
                          </>
                        ) : null}
                        {item.item_type === 'document' ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => navigate(`/search?q=${encodeURIComponent(item.title || item.item_key)}`)}
                          >
                            <Search className="w-4 h-4 mr-1" />
                            Search
                          </Button>
                        ) : null}
                        {item.url ? (
                          <a
                            href={item.url}
                            target="_blank"
                            rel="noreferrer"
                            className="text-sm text-primary-600 hover:text-primary-700 flex items-center gap-1 justify-center"
                          >
                            <Link2 className="w-4 h-4" />
                            Open
                          </a>
                        ) : null}
                      </div>
                    </div>
                    {item.feedback ? (
                      <div className="mt-3 text-xs text-gray-600 bg-gray-50 border border-gray-100 rounded p-2">
                        Feedback: {item.feedback}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Modals */}
      {showCreateModal && (
        <CreateJobModal
          onClose={() => setShowCreateModal(false)}
          createMutation={createMutation}
        />
      )}
      {showClaudeQuickStartModal && (
        <QuickStartClaudeBackendModal
          onClose={() => setShowClaudeQuickStartModal(false)}
          quickStartClaudeBackendMutation={quickStartClaudeBackendMutation}
          codeSources={codeSources}
          templateRecommendGoal={templateRecommendGoal}
        />
      )}
      {showDomainResearchQuickStartModal && (
        <QuickStartDomainResearchModal
          onClose={() => setShowDomainResearchQuickStartModal(false)}
          quickStartDomainResearchMutation={quickStartDomainResearchMutation}
          codeSources={codeSources}
          templateRecommendGoal={templateRecommendGoal}
        />
      )}
      {activeSwarmQuickStart && (
        <QuickStartCodingSwarmModal
          codeSources={codeSources}
          codingSwarmProfiles={codingSwarmProfiles}
          createCodingSwarmProfileMutation={createCodingSwarmProfileMutation}
          updateCodingSwarmProfileMutation={updateCodingSwarmProfileMutation}
          deleteCodingSwarmProfileMutation={deleteCodingSwarmProfileMutation}
          currentUserId={String(user?.id || '')}
          presetKey={activeSwarmQuickStart.preset.presetKey}
          title={activeSwarmQuickStart.preset.title}
          description={activeSwarmQuickStart.preset.description}
          defaultName={`${activeSwarmQuickStart.preset.namePrefix} - ${new Date().toLocaleDateString()}`}
          defaultFailureSymptom={
            templateRecommendGoal.trim() ||
            activeSwarmQuickStart.preset.failureSymptomPlaceholder
          }
          defaultGoal={activeSwarmQuickStart.preset.defaultGoal}
          defaultScope={activeSwarmQuickStart.preset.defaultScope}
          accentClassName={activeSwarmQuickStart.preset.accentClassName}
          initialProfileId={
            codingSwarmLaunchSeed?.presetKey === activeSwarmQuickStart.preset.presetKey
              ? codingSwarmLaunchSeed?.profileId
              : undefined
          }
          initialSourceId={
            codingSwarmLaunchSeed?.presetKey === activeSwarmQuickStart.preset.presetKey
              ? codingSwarmLaunchSeed?.sourceId
              : undefined
          }
          onClose={() => {
            activeSwarmQuickStart.setOpen(false);
            setCodingSwarmLaunchSeed(null);
          }}
          submitLabel={activeSwarmQuickStart.preset.submitLabel}
          submitMutation={activeSwarmQuickStart.mutation}
          buildPayload={activeSwarmQuickStart.buildPayload}
        />
      )}
      {showRepoBugTriageQuickStartModal && (
        <QuickStartRepoBugTriageModal
          onClose={() => setShowRepoBugTriageQuickStartModal(false)}
          quickStartRepoBugTriageMutation={quickStartRepoBugTriageMutation}
          codeSources={codeSources}
          templateRecommendGoal={templateRecommendGoal}
        />
      )}
      {showRoleWorkflowQuickStartModal && (
        <QuickStartRoleWorkflowModal
          onClose={() => setShowRoleWorkflowQuickStartModal(false)}
          quickStartRoleWorkflowMutation={quickStartRoleWorkflowMutation}
          templateRecommendGoal={templateRecommendGoal}
        />
      )}
      {createFromTemplate && (
        <CreateFromTemplateModal
          template={createFromTemplate}
          onClose={() => setCreateFromTemplate(null)}
          createFromTemplateMutation={createFromTemplateMutation}
          codeSources={codeSources}
        />
      )}
      {showCustomerResearchModal && (
        <CustomerResearchModal
          onClose={() => setShowCustomerResearchModal(false)}
          createFromTemplateMutation={createFromTemplateMutation}
          createFromChainMutation={createFromChainMutation}
          templatesData={templatesData}
          chainsData={chainsData}
        />
      )}
      {showInboxMonitorModal && (
        <InboxMonitorModal
          onClose={() => setShowInboxMonitorModal(false)}
          createInboxMonitorMutation={createInboxMonitorMutation}
        />
      )}
      {showMonitorProfilesModal && (
        <MonitorProfilesModal
          onClose={() => setShowMonitorProfilesModal(false)}
          monitorProfiles={monitorProfiles}
          monitorProfilesLoading={monitorProfilesLoading}
          refetchMonitorProfiles={refetchMonitorProfiles}
          upsertMonitorProfileMutation={upsertMonitorProfileMutation}
        />
      )}
      {startFromChain && (
        <StartChainModal
          chain={startFromChain}
          onClose={() => setStartFromChain(null)}
          createFromChainMutation={createFromChainMutation}
        />
      )}

      {/* Chain Status Modal */}
      {selectedChainStatus && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
            <div className="p-4 border-b border-gray-200 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
                  <GitBranch className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold">Chain Status</h2>
                  <p className="text-sm text-gray-500">
                    Step {selectedChainStatus.current_step + 1} of {selectedChainStatus.total_steps} •{' '}
                    <span className="capitalize">{selectedChainStatus.status}</span>
                  </p>
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setSelectedChainStatus(null)}>
                <XCircle className="w-5 h-5" />
              </Button>
            </div>

            {chainExperimentStopInfo ? (
              <div className="px-4 py-3 border-b border-gray-200 bg-amber-50">
                <div className="flex items-start justify-between gap-3">
                  <div className="text-sm text-amber-900">
                    <div className="font-medium">Experiment loop stopped early</div>
                    <div className="text-xs text-amber-800 mt-1">
                      Reason: <span className="font-mono">{chainExperimentStopInfo.reason || 'unknown'}</span>
                      {chainExperimentStopInfo.atRunId ? (
                        <>
                          {' '}
                          • Run: <span className="font-mono">{chainExperimentStopInfo.atRunId}</span>
                        </>
                      ) : null}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {chainExperimentStopInfo.noteId ? (
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => {
                          setSelectedChainStatus(null);
                          navigate(`/research-notes?note=${encodeURIComponent(chainExperimentStopInfo.noteId || '')}`);
                        }}
                      >
                        Open note
                      </Button>
                    ) : null}
                    {chainExperimentStopInfo.stoppedByJobId ? (
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => {
                          setSelectedChainStatus(null);
                          navigate(buildAutonomousAgentsUrl(String(chainExperimentStopInfo.stoppedByJobId || '')));
                        }}
                      >
                        Open job
                      </Button>
                    ) : null}
                  </div>
                </div>
              </div>
            ) : null}

            {/* Progress bar */}
            <div className="px-4 py-3 border-b border-gray-200">
              <div className="flex items-center justify-between text-sm text-gray-500 mb-1">
                <span>Overall Progress</span>
                <span>{selectedChainStatus.overall_progress}%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    selectedChainStatus.status === 'completed'
                      ? 'bg-green-500'
                      : selectedChainStatus.status === 'failed'
                      ? 'bg-red-500'
                      : 'bg-purple-500'
                  }`}
                  style={{ width: `${selectedChainStatus.overall_progress}%` }}
                />
              </div>
            </div>

            {/* Jobs list */}
            <div className="flex-1 overflow-y-auto p-4">
              <div className="space-y-3">
                {selectedChainStatus.jobs.map((job, index) => {
                  const statusConfig = STATUS_CONFIG[job.status as AgentJobStatus] || STATUS_CONFIG.pending;
                  const StatusIcon = statusConfig.icon;
                  const isCurrentStep = index === selectedChainStatus.current_step;

                  return (
                    <div
                      key={job.id}
                      className={`border rounded-lg p-3 ${
                        isCurrentStep ? 'border-purple-500 bg-purple-50' : 'border-gray-200'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="flex items-center justify-center w-6 h-6 rounded-full bg-gray-200 text-xs font-medium">
                            {index + 1}
                          </div>
                          <div>
                            <h4 className="font-medium text-gray-900">{job.name}</h4>
                            <p className="text-xs text-gray-500">{job.job_type}</p>
                          </div>
                        </div>
                        <div
                          className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.color}`}
                        >
                          <StatusIcon className={`w-3 h-3 ${job.status === 'running' ? 'animate-spin' : ''}`} />
                          <span className="capitalize">{job.status}</span>
                        </div>
                      </div>

                      {/* Mini progress bar */}
                      <div className="mt-2">
                        <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              job.status === 'completed'
                                ? 'bg-green-500'
                                : job.status === 'failed'
                                ? 'bg-red-500'
                                : 'bg-purple-500'
                            }`}
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                      </div>

                      {job.error && (
                        <p className="mt-2 text-xs text-red-600 bg-red-50 rounded p-2">{job.error}</p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="p-4 border-t border-gray-200 flex justify-end">
              <Button variant="secondary" onClick={() => setSelectedChainStatus(null)}>
                Close
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Export Modal */}
      {showExportModal && exportingJob && (
        <ExportModal
          job={exportingJob}
          onClose={() => {
            setShowExportModal(false);
            setExportingJob(null);
          }}
        />
      )}
    </div>
  );
};

// Export Modal Component
const ExportModal: React.FC<{ job: AgentJob; onClose: () => void }> = ({ job, onClose }) => {
  const [format, setFormat] = useState<'docx' | 'pdf' | 'pptx'>('docx');
  const [style, setStyle] = useState<'professional' | 'technical' | 'casual'>('professional');
  const [includeLog, setIncludeLog] = useState(false);
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [enhance, setEnhance] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async () => {
    setIsExporting(true);
    try {
      await apiClient.downloadJobExport(job.id, job.name, format, {
        style,
        includeLog,
        includeMetadata,
        enhance,
      });
      toast.success(`Exported as ${format.toUpperCase()}${enhance ? ' (AI-enhanced)' : ''}`);
      onClose();
    } catch (error: any) {
      console.error('Export failed:', error);
      toast.error(error.message || 'Export failed');
    } finally {
      setIsExporting(false);
    }
  };

  const formatOptions = [
    { value: 'docx', label: 'Word Document', icon: FileText, description: 'DOCX format, editable' },
    { value: 'pdf', label: 'PDF Document', icon: FileText, description: 'PDF format, universal' },
    { value: 'pptx', label: 'Presentation', icon: FileDown, description: 'PowerPoint slides' },
  ];

  const styleOptions = [
    { value: 'professional', label: 'Professional', description: 'Clean corporate look' },
    { value: 'technical', label: 'Technical', description: 'Developer-focused' },
    { value: 'casual', label: 'Casual', description: 'Friendly and approachable' },
  ];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
        <div className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-blue-100 text-blue-600">
              <Download className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Export Results</h2>
              <p className="text-sm text-gray-500">Export "{job.name}" results</p>
            </div>
          </div>

          <div className="space-y-4">
            {/* Format selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Format</label>
              <div className="grid grid-cols-3 gap-2">
                {formatOptions.map((option) => {
                  const Icon = option.icon;
                  return (
                    <button
                      key={option.value}
                      className={`p-3 border rounded-lg text-center transition-colors ${
                        format === option.value
                          ? 'border-primary-500 bg-primary-50 text-primary-700'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setFormat(option.value as any)}
                    >
                      <Icon className="w-5 h-5 mx-auto mb-1" />
                      <span className="text-xs font-medium block">{option.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Style selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Style</label>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={style}
                onChange={(e) => setStyle(e.target.value as any)}
              >
                {styleOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label} - {option.description}
                  </option>
                ))}
              </select>
            </div>

            {/* Options */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">Options</label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={includeMetadata}
                  onChange={(e) => setIncludeMetadata(e.target.checked)}
                  className="rounded"
                />
                <span>Include job metadata and statistics</span>
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={includeLog}
                  onChange={(e) => setIncludeLog(e.target.checked)}
                  className="rounded"
                />
                <span>Include execution log</span>
              </label>
            </div>

            {/* AI Enhancement */}
            <div className="border border-purple-200 rounded-lg p-3 bg-purple-50">
              <label className="flex items-start gap-3">
                <input
                  type="checkbox"
                  checked={enhance}
                  onChange={(e) => setEnhance(e.target.checked)}
                  className="rounded mt-0.5 border-purple-300"
                />
                <div>
                  <span className="text-sm font-medium text-purple-900 flex items-center gap-1">
                    <Zap className="w-4 h-4" />
                    AI-Enhanced Report
                  </span>
                  <p className="text-xs text-purple-700 mt-0.5">
                    Uses AI to generate an executive summary, key insights, and recommendations.
                    Takes longer to generate.
                  </p>
                </div>
              </label>
            </div>

            {/* Job summary */}
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-1">Export preview</p>
              <div className="text-sm space-y-1">
                <p><span className="text-gray-500">Status:</span> {job.status}</p>
                <p><span className="text-gray-500">Progress:</span> {job.progress}%</p>
                {job.results?.findings_count !== undefined && (
                  <p><span className="text-gray-500">Findings:</span> {job.results.findings_count}</p>
                )}
              </div>
            </div>
          </div>

          <div className="flex justify-end gap-3 mt-6 pt-4 border-t">
            <Button variant="secondary" onClick={onClose} disabled={isExporting}>
              Cancel
            </Button>
            <Button onClick={handleExport} disabled={isExporting}>
              {isExporting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Export {format.toUpperCase()}
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AutonomousAgentsPage;
