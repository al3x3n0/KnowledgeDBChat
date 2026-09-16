/**
 * Autonomous Agents Page
 *
 * Manage and monitor autonomous agent jobs that run independently
 * to accomplish goals like research, monitoring, and analysis.
 */

import React, { Suspense, lazy, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Activity,
  AlertCircle,
  BarChart3,
  Bot,
  Brain,
  Bug,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Download,
  Eye,
  FileCheck,
  FileDown,
  FileText,
  Filter,
  GitBranch,
  Inbox,
  Layers,
  ListChecks,
  Loader2,
  Map as MapIcon,
  PanelRightClose,
  PanelRightOpen,
  Play,
  Plus,
  RefreshCw,
  Rocket,
  RotateCcw,
  Search,
  Settings,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Users,
  XCircle,
  Zap,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import type {
  User,
  AgentCheckpointQueueItem,
  AgentCheckpointQueueAction,
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
  CollaborationSummary,
  CodingBacklogItem,
  CodingBacklogItemCreate,
  CodingSwarmProfile,
  CodingSwarmProfileCreate,
  CodingSwarmProfileUpdate,
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
  AgentJobChainDefinition,
  AgentJobChainStatus,
  AgentJobFromChainCreate,
  ResearchInboxItem,
  ResearchInboxItemStatus,
  ResearchMonitorAnalyticsResponse,
  ResearchMonitorCustomerRebalanceEvaluationDetail,
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
import {
  formatQueueHealthDrilldownLabel,
  normalizeInboxHealthDrilldown,
  normalizeInboxPolicyDrilldown,
  normalizeQueueHealthDrilldown,
} from '../components/agent/drilldowns';
import type {
  InboxHealthDrilldown,
  InboxPolicyDrilldown,
  QueueHealthDrilldown,
} from '../components/agent/drilldowns';
import SkeletonList from '../components/common/SkeletonList';
import CreateFromTemplateModal from '../components/agent/CreateFromTemplateModal';
import NewCampaignModal from '../components/agent/NewCampaignModal';
import CollaborationSummaryPanel from '../components/agent/CollaborationSummaryPanel';

// what extraction was *for*; it just is not what extraction *is*.
import { invalidateAgentRunQueries } from '../utils/agentRunQueries';
import type { QuickStart } from '../components/agent/tabs/JobTemplatesTab';
import PluginSlot from '../plugins/PluginSlot';
import InboxMonitorModal from '../components/agent/InboxMonitorModal';
import MonitorProfilesModal from '../components/agent/MonitorProfilesModal';
import QuickStartClaudeBackendModal from '../components/agent/QuickStartClaudeBackendModal';
import QuickStartCodingSwarmModal from '../components/agent/QuickStartCodingSwarmModal';
import QuickStartRepoBugTriageModal from '../components/agent/QuickStartRepoBugTriageModal';
import QuickStartRoleWorkflowModal from '../components/agent/QuickStartRoleWorkflowModal';
import QuickStartDomainResearchModal from '../components/agent/QuickStartDomainResearchModal';
import CreateJobModal from '../components/agent/CreateJobModal';
import StartChainModal from '../components/agent/StartChainModal';
import {
  STATUS_CONFIG,
  type AgentJobsTab,
} from '../components/agent/jobConfig';
import { getLatestExperimentRun } from '../components/agent/jobFields';
import JobCard from '../components/agent/JobCard';
import JobDetailPanel from '../components/agent/JobDetailPanel';
import SwarmOutcomesPanel from '../components/agent/SwarmOutcomesPanel';
import useSwarmOutcomes from '../components/agent/useSwarmOutcomes';
import {
  humanizeDecisionTraceValue,
  summarizeSchedulerState,
} from '../utils/agentJobDetail';
import { swarmQuickStartPreset } from '../components/agent/swarmQuickStarts';
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

// Tabs load when their tab is opened, not when the page is.
//
// Extraction alone did not shrink the bundle -- the components landed in the
// same chunk, so opening "My Jobs" still downloaded Swarm Review. Splitting
// is what extraction was *for*; it just is not what extraction *is*.
const JobChainsTab = lazy(() => import('../components/agent/tabs/JobChainsTab'));
const SwarmReviewTab = lazy(() => import('../components/agent/tabs/SwarmReviewTab'));
const ResearchInboxTab = lazy(() => import('../components/agent/tabs/ResearchInboxTab'));
const CodingBacklogTab = lazy(() => import('../components/agent/tabs/CodingBacklogTab'));
const DecisionTraceTab = lazy(() => import('../components/agent/tabs/DecisionTraceTab'));
const AutonomyHealthTab = lazy(() => import('../components/agent/tabs/AutonomyHealthTab'));
const JobTemplatesTab = lazy(() => import('../components/agent/tabs/JobTemplatesTab'));

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



const AUTONOMY_FOCUS_ROW_CLASS = 'border-cyan-300 bg-cyan-50 ring-2 ring-cyan-200';
const AUTONOMY_FOCUS_CARD_CLASS = 'border-cyan-300 ring-2 ring-cyan-200';





const scientificValidationStatusClasses = (status?: string | null) => {
  const normalized = String(status || '').trim().toLowerCase();
  if (normalized === 'blocked' || normalized === 'failed') return 'bg-rose-100 text-rose-700';
  if (normalized === 'running' || normalized === 'paused' || normalized === 'provisioning' || normalized === 'queued') return 'bg-amber-100 text-amber-800';
  if (normalized === 'succeeded' || normalized === 'completed') return 'bg-emerald-100 text-emerald-700';
  return 'bg-gray-200 text-gray-700';
};

const synthesisStatusClasses = (status?: string | null) => {
  const normalized = String(status || '').trim().toLowerCase();
  if (normalized === 'completed') return 'bg-emerald-100 text-emerald-700';
  if (normalized === 'failed' || normalized === 'cancelled') return 'bg-rose-100 text-rose-700';
  if (normalized) return 'bg-amber-100 text-amber-800';
  return 'bg-gray-200 text-gray-700';
};



const researchOpportunityStageClass = (value?: string | null) => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'completed') return 'bg-emerald-100 text-emerald-700';
  if (normalized === 'blocked' || normalized === 'suppressed') return 'bg-rose-100 text-rose-700';
  if (normalized === 'planned' || normalized === 'accepted') return 'bg-blue-100 text-blue-700';
  if (normalized === 'validating') return 'bg-amber-100 text-amber-800';
  return 'bg-gray-200 text-gray-700';
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
      : 'bg-gray-200 text-gray-700';
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-gray-600">
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
    ? 'bg-gray-200 text-gray-700'
    : 'bg-violet-100 text-violet-700';
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-gray-600">
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

/**
 * The tabs, grouped by the activity they belong to.
 *
 * Thirteen flat tabs asked the user to hold the whole taxonomy in their head:
 * "Swarm Review" and "Swarm Outcomes" and "Swarm Profiles" sat adjacent and
 * are three different activities — approving something, reading what happened,
 * and configuring for next time.
 *
 * `urgent` marks the groups whose counts mean "this is waiting for you", which
 * is the only kind of count worth colouring.
 */
const TAB_GROUPS: Array<{
  name: string;
  tabs: Array<{
    id: AgentJobsTab;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    urgent?: boolean;
  }>;
}> = [
  {
    name: 'Work',
    tabs: [
      { id: 'jobs', label: 'My Jobs', icon: Bot },
      { id: 'chains', label: 'Job Chains', icon: GitBranch },
      { id: 'templates', label: 'Templates', icon: FileCheck },
    ],
  },
  {
    name: 'Needs you',
    tabs: [
      { id: 'queue', label: 'Checkpoints', icon: AlertCircle, urgent: true },
      { id: 'inbox', label: 'Inbox', icon: Inbox, urgent: true },
      { id: 'swarm', label: 'Swarm Review', icon: Users, urgent: true },
    ],
  },
  {
    name: 'What happened',
    tabs: [
      { id: 'trace', label: 'Decision Trace', icon: Clock },
      { id: 'health', label: 'Autonomy Health', icon: Activity },
      { id: 'outcomes', label: 'Swarm Outcomes', icon: BarChart3 },
    ],
  },
  {
    name: 'Portfolios',
    tabs: [
      { id: 'fleet', label: 'Research Fleet', icon: Sparkles },
      { id: 'backlog', label: 'Coding Backlog', icon: ListChecks },
    ],
  },
  {
    name: 'Setup',
    tabs: [
      { id: 'profiles', label: 'Swarm Profiles', icon: Settings },
      { id: 'domain', label: 'Domain Profiles', icon: Brain },
    ],
  },
];


const AutonomousAgentsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'queue' | 'trace' | 'health' | 'jobs' | 'swarm' | 'outcomes' | 'profiles' | 'templates' | 'chains' | 'inbox' | 'backlog' | 'domain' | 'fleet' | 'create'>('jobs');
  const [showSystemMap, setShowSystemMap] = useState(false);
  const [selectedJob, setSelectedJob] = useState<AgentJob | null>(null);
  // The jobs filter row holds twelve controls and is collapsed by default.
  // Most visits to this page are "what is running", not "narrow this down",
  // and twelve controls charged against every one of those visits is the bulk
  // of why the page reads as cluttered.
  const [showJobFilters, setShowJobFilters] = useState<boolean>(false);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [launchModeFilter, setLaunchModeFilter] = useState<string>('');
  const [hasRelaunchChildrenFilter, setHasRelaunchChildrenFilter] = useState<string>('');
  const [relaunchFromJobIdFilter, setRelaunchFromJobIdFilter] = useState<string>('');
  // The detail panel collapses, and remembers it. It is a third of the width
  // and permanently open, which is a lot of screen to give a panel you are not
  // reading. localStorage is wrapped because it throws outright in a private
  // window rather than returning null.
  const [detailPanelCollapsed, setDetailPanelCollapsed] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem('agent_detail_panel_collapsed') === '1';
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      window.localStorage.setItem(
        'agent_detail_panel_collapsed',
        detailPanelCollapsed ? '1' : '0'
      );
    } catch {
      // Still works, just does not persist.
    }
  }, [detailPanelCollapsed]);

  const [swarmOnlyFilter, setSwarmOnlyFilter] = useState<boolean>(false);
  const [swarmSortBy, setSwarmSortBy] = useState<string>('created_desc');
  const [swarmMinConsensus, setSwarmMinConsensus] = useState<number>(0);
  const [graphHealthFilter, setGraphHealthFilter] = useState<string>('');
  const [graphSortBy, setGraphSortBy] = useState<string>('none');
  const [dedupSkipFilter, setDedupSkipFilter] = useState<string>('');
  const [scopeGuardFilter, setScopeGuardFilter] = useState<string>('');
  const [experimentRecoveryFilter, setExperimentRecoveryFilter] = useState<string>('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showNewCampaignModal, setShowNewCampaignModal] = useState(false);
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
  const swarmOutcomes = useSwarmOutcomes(activeTab);
  const {
    swarmOutcomeBySwarmJobId,
    swarmOutcomeByRepairJobId,
  } = swarmOutcomes;
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
  const [backlogFailureSymptom, setBacklogFailureSymptom] = useState('');
  const [backlogCommandsText, setBacklogCommandsText] = useState('');
  const [backlogFilePathsText, setBacklogFilePathsText] = useState('');
  const [backlogVisibilityScope, setBacklogVisibilityScope] = useState<'mine' | 'shared' | 'all'>('mine');
  const [backlogAssignmentFilter, setBacklogAssignmentFilter] = useState<string>('');
  const [backlogQueueStateFilter, setBacklogQueueStateFilter] = useState<string>('');
  const [backlogNoteDrafts, setBacklogNoteDrafts] = useState<Record<string, string>>({});
  const [backlogCloseReasonDrafts, setBacklogCloseReasonDrafts] = useState<Record<string, string>>({});
  // Stays on the page: it is a query key. Both the swarm-review job list
  // and the analytics refetch when it changes, and those queries live here.
  const [swarmReviewVisibilityScope, setSwarmReviewVisibilityScope] =
    useState<'mine' | 'shared' | 'all'>('mine');
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
  const [inboxBulkRejectReason, setInboxBulkRejectReason] = useState<string>('');
  const [inboxBulkFollowUpNote, setInboxBulkFollowUpNote] = useState<string>('');
  const [inboxRejectReasonDrafts, setInboxRejectReasonDrafts] = useState<Record<string, string>>({});
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
      // Loaded whether or not this tab is open: the badge exists to answer
      // "is anything waiting for me?" from wherever you are, and a count that
      // only appears once you have clicked the tab cannot do that. Polled
      // slowly when you are elsewhere.
      enabled: true,
      refetchInterval: activeTab === 'queue' ? 10000 : 60000,
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


  const selectedTraceView = useMemo(
    () => (traceViewsData?.items || []).find((item) => item.id === selectedTraceViewId) || null,
    [selectedTraceViewId, traceViewsData?.items]
  );


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
          invalidateAgentRunQueries(queryClient, [
            'research-portfolios',
            'domain-research-profiles',
            'research-inbox',
          ]);
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
  // The seven quick starts were seven near-identical JSX blocks differing in
  // three values. Declared once here; the tab renders whatever it is given
  // and knows nothing about scopes or modals.
  const templateQuickStarts: QuickStart[] = [
    {
      label: 'Start Domain Research',
      onStart: () => {
        setTemplateRecommendScope('research');
        if (!templateRecommendGoal.trim()) {
          setTemplateRecommendGoal(
            'Research a technical domain, rank evidence-backed ideas, and generate notes'
          );
        }
        setShowDomainResearchQuickStartModal(true);
      },
    },
    {
      label: 'Start Bug Triage Swarm',
      onStart: () => {
        setTemplateRecommendScope('repo');
        if (!templateRecommendGoal.trim()) {
          setTemplateRecommendGoal(
            'Run a coding swarm to reproduce the bug, rank the best repair path, and auto-launch the repair loop'
          );
        }
        setShowBugTriageSwarmQuickStartModal(true);
      },
    },
    {
      label: 'Start Build Break Swarm',
      onStart: () => {
        setTemplateRecommendScope('backend');
        if (!templateRecommendGoal.trim()) {
          setTemplateRecommendGoal(
            'Diagnose the build break, isolate the failing file cluster, and auto-handoff the winning repair path'
          );
        }
        setShowBuildBreakSwarmQuickStartModal(true);
      },
    },
    {
      label: 'Start Frontend Regression Swarm',
      onStart: () => {
        setTemplateRecommendScope('frontend');
        if (!templateRecommendGoal.trim()) {
          setTemplateRecommendGoal(
            'Reproduce the frontend regression, isolate the affected UI surface, and promote the winning repair path'
          );
        }
        setShowFrontendRegressionSwarmQuickStartModal(true);
      },
    },
    {
      label: 'Start Repo Bug Triage',
      onStart: () => {
        setTemplateRecommendScope('repo');
        if (!templateRecommendGoal.trim()) {
          setTemplateRecommendGoal(
            'Triage a repo bug from the observed symptom and return a verified patch proposal'
          );
        }
        setShowRepoBugTriageQuickStartModal(true);
      },
    },
    {
      label: 'Start Claude Backend Loop',
      onStart: () => {
        setTemplateRecommendScope('backend');
        if (!templateRecommendGoal.trim()) {
          setTemplateRecommendGoal(
            'Fix backend API tests and stabilize integrations'
          );
        }
        setShowClaudeQuickStartModal(true);
      },
    },
    {
      label: 'Start Role Workflow',
      onStart: () => {
        if (!templateRecommendGoal.trim()) {
          setTemplateRecommendGoal(
            'Investigate contradictory signals and produce a validated recommendation plan'
          );
        }
        setShowRoleWorkflowQuickStartModal(true);
      },
    },
  ];

const formatAutonomyLabel = (value?: string | null) => String(value || 'balanced').replace(/_/g, ' ');

const formatReviewModeLabel = (value?: string | null) => String(value || 'auto_launch_safe').replace(/_/g, ' ');

  const [healthMonitorTypeFilter, setHealthMonitorTypeFilter] = useState<string>('');

  const [healthBucketFilter, setHealthBucketFilter] = useState<string>('');

  const [healthAutonomyFilter, setHealthAutonomyFilter] = useState<string>('');

  const { data: monitorAnalyticsData, isLoading: monitorAnalyticsLoading, refetch: refetchMonitorAnalytics } = useQuery(
    ['research-monitor-analytics'],
    () => apiClient.getResearchMonitorAnalytics(),
    {
      // Health *and* inbox. `healthCustomers` is derived from this response and
      // fills the inbox's customer filter, so gating it on the health tab left
      // that dropdown empty unless you had opened Autonomy Health earlier in
      // the same session -- a filter offering nothing, depending on where you
      // had been.
      enabled: activeTab === 'health' || activeTab === 'inbox',
      staleTime: 30000,
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
        invalidateAgentRunQueries(queryClient, [
          'research-monitor-analytics',
          'research-inbox',
        ]);
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
        invalidateAgentRunQueries(queryClient, [
          'research-monitor-analytics',
          'research-inbox',
        ]);
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

  const [healthPolicySimulations, setHealthPolicySimulations] = useState<Record<string, ResearchMonitorPolicySimulationResponse>>({});
  const [healthPolicyEvaluations, setHealthPolicyEvaluations] = useState<Record<string, ResearchMonitorPolicyEvaluationDetail>>({});
  const [healthCustomerRebalanceEvaluations, setHealthCustomerRebalanceEvaluations] = useState<Record<string, ResearchMonitorCustomerRebalanceEvaluationDetail>>({});

  // Written by a page-resident mutation *and* read by the health tab, so it
  // lives here and goes down as a prop. Moving it into the tab left the page's
  // mutation writing one state while the tab read another -- two states with
  // the same name, and no error until the build.
  const [healthPolicyDrafts, setHealthPolicyDrafts] = useState<Record<string, { automation_profile: string; mode: string; allowed: string[] }>>({});

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
                  <div className="mt-1 text-gray-600">
                    Latest action: {latestOperatorAction}
                    {latestOperatorOutcome ? ` · ${latestOperatorOutcome}` : ''}
                  </div>
                ) : null}
                {Number(run.retry_count || 0) > 0 || run.parent_run_id || run.latest_child_run_id ? (
                  <div className="mt-1 text-gray-500">
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
        invalidateAgentRunQueries(queryClient);
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
        invalidateAgentRunQueries(queryClient);
        toast.success('Job deleted');
        setSelectedJob(null);
        navigate(buildAutonomousAgentsUrl(), { replace: true });
      },
      onError: (error: any) => {
        toast.error(error.message || 'Delete failed');
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






  const createMutation = useMutation(
    (data: AgentJobCreate) => apiClient.createAgentJob(data),
    {
      onSuccess: (job) => {
        invalidateAgentRunQueries(queryClient);
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
        invalidateAgentRunQueries(queryClient, [
          'domain-research-profiles',
        ]);
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
          invalidateAgentRunQueries(queryClient, [
            'research-inbox',
            'research-inbox-stats',
            'agent-decision-trace',
            'agent-decision-trace-analytics',
            'notifications',
            'notifications-unread-count',
          ]);
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
          invalidateAgentRunQueries(queryClient, [
            'research-inbox',
            'research-inbox-stats',
            'agent-decision-trace',
            'agent-decision-trace-analytics',
            'notifications',
            'notifications-unread-count',
          ]);
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
      <div className="mt-2 rounded border border-gray-200 bg-gray-100 p-2 text-[11px] text-gray-700">
        <div className="flex items-center justify-between gap-2">
          <div className="font-medium text-gray-800">{resolveOpportunityExplanationHeading(row)}</div>
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
                <span className="font-medium text-gray-800">{item.label}:</span>{' '}
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


  const createInboxMonitorMutation = useMutation(
    (data: AgentJobCreate) => apiClient.createAgentJob(data),
    {
      onSuccess: (job) => {
        invalidateAgentRunQueries(queryClient);
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
        invalidateAgentRunQueries(queryClient);
        toast.success('Job created from template');
        setCreateFromTemplate(null);
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
        invalidateAgentRunQueries(queryClient);
        toast.success('Chain started');
        setStartFromChain(null);
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
        invalidateAgentRunQueries(queryClient, [
          'research-inbox',
          'research-inbox-stats',
          'research-portfolios',
          'domain-research-profiles',
        ]);
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
        invalidateAgentRunQueries(queryClient, [
          'research-inbox',
          'research-inbox-stats',
          'research-portfolios',
          'domain-research-profiles',
        ]);
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
        invalidateAgentRunQueries(queryClient, [
          'research-portfolios',
          'domain-research-profiles',
          'research-inbox',
          'research-inbox-stats',
          'agent-decision-trace',
          'agent-decision-trace-analytics',
          'notifications',
          'notifications-unread-count',
        ]);
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
        invalidateAgentRunQueries(queryClient);
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
        invalidateAgentRunQueries(queryClient, [
          'research-inbox',
          'research-inbox-stats',
          'research-portfolios',
          'domain-research-profiles',
          'agent-decision-trace',
          'agent-decision-trace-analytics',
        ]);
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
                <div className="mt-1 text-[11px] text-gray-500">
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
                <div className="mt-1 text-[11px] text-gray-500">
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
        <div className="rounded border border-gray-200 bg-gray-100 p-2 space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs text-gray-600">
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


  const { data: myPreferences } = useQuery(['me-preferences'], () => apiClient.getMyPreferences(), {
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });

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

  // Every badge in the tab bar, resolved once. A count that only exists when
  // its own tab is open cannot answer "is anything waiting for me?", which is
  // the question the bar is there to answer.
  const tabCounts: Partial<Record<AgentJobsTab, number>> = {
    queue: checkpointQueueData?.total || 0,
    trace: decisionTraceData?.total || 0,
    inbox: inboxStats?.new || 0,
  };

  /** How many of the jobs-list filters are actually narrowing anything.
   *
   *  Shown on the collapsed disclosure so a hidden filter can never silently
   *  shape the list: the reason to hide the controls is that they are usually
   *  unused, and the moment one IS used it has to announce itself. A filtered
   *  list that looks unfiltered is worse than a cluttered toolbar.
   *
   *  `graphSortBy` counts only when it is not 'none' -- a sort is not a filter,
   *  but a non-default one still changes what you see first.
   */
  const activeJobFilterCount = useMemo(() => {
    const values = [
      statusFilter,
      typeFilter,
      launchModeFilter,
      hasRelaunchChildrenFilter,
      relaunchFromJobIdFilter,
      graphHealthFilter,
      dedupSkipFilter,
      scopeGuardFilter,
      experimentRecoveryFilter,
    ];
    let count = values.filter((v) => String(v || '').trim()).length;
    if (swarmOnlyFilter) count += 1;
    if (graphSortBy && graphSortBy !== 'none') count += 1;
    return count;
  }, [
    statusFilter,
    typeFilter,
    launchModeFilter,
    hasRelaunchChildrenFilter,
    relaunchFromJobIdFilter,
    graphHealthFilter,
    dedupSkipFilter,
    scopeGuardFilter,
    experimentRecoveryFilter,
    swarmOnlyFilter,
    graphSortBy,
  ]);

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
                  <span className={`text-[11px] px-2 py-0.5 rounded ${profile.enabled ? 'bg-emerald-100 text-emerald-700' : 'bg-gray-200 text-gray-600'}`}>
                    {profile.enabled ? 'enabled' : 'disabled'}
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-indigo-100 text-indigo-700">
                    {String(profile.track_type || 'generic')}
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-gray-200 text-gray-700">
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
          <Button variant="secondary" onClick={() => setShowNewCampaignModal(true)}>
            <Rocket className="w-4 h-4 mr-2" />
            New Campaign
          </Button>
          <Button onClick={() => setShowCreateModal(true)}>
            <Plus className="w-4 h-4 mr-2" />
            New Job
          </Button>
        </div>
      </div>

      {/* Anything an installed plugin contributes to this page. */}
      <PluginSlot slot="runs.header" compact={false} />

      {showSystemMap && (
        <div className="mb-6 rounded-lg border border-gray-200 bg-gray-100 p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-sm font-semibold text-gray-900">System Map</h2>
              <p className="mt-1 text-sm text-gray-600">
                Current operator surface and runtime ownership. Canonical doc: <code>docs/ARCHITECTURE_ASCII.md</code>
              </p>
            </div>
            <span className="rounded bg-white px-2 py-1 text-xs text-gray-500 border border-gray-200">
              Canonical autonomy: <code>automation_profile</code> / <code>automation_policy</code> / <code>effective_policy</code>
            </span>
          </div>
          <pre className="mt-4 overflow-x-auto rounded border border-gray-200 bg-white p-3 text-xs leading-5 text-gray-700">
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
        {/* The tabs, grouped by what you are doing.
            Thirteen of them in one flat row mixed four different activities:
            work to start, work waiting on you, understanding a run that
            happened, and configuration. That is the same flat-list problem the
            navigation had at thirty-one destinations — the grouping existed
            and was invisible.

            The counts also used to load only once you were already on the tab,
            so "is anything waiting for me?" could not be answered without
            clicking three of them. They load regardless now, slowly when you
            are elsewhere. */}
        <div className="mb-4 border-b border-gray-200">
          <div className="flex flex-wrap items-end gap-x-5 gap-y-1">
            {TAB_GROUPS.map((group, groupIndex) => (
              <div key={group.name} className="flex items-end gap-1">
                {groupIndex > 0 && (
                  <span
                    className="mx-1 mb-2 h-4 w-px bg-gray-300"
                    aria-hidden="true"
                  />
                )}
                <div className="flex flex-col">
                  <span className="px-1 text-[10px] font-semibold uppercase tracking-wide text-gray-500">
                    {group.name}
                  </span>
                  <div className="flex gap-1">
                    {group.tabs.map((tab) => {
                      const Icon = tab.icon;
                      const count = tabCounts[tab.id] || 0;
                      const active = activeTab === tab.id;
                      return (
                        <button
                          key={tab.id}
                          type="button"
                          aria-current={active ? 'page' : undefined}
                          className={`pb-2 px-2 text-sm font-medium flex items-center gap-1.5 border-b-2
                            transition-colors duration-fast ${
                              active
                                ? 'text-primary-600 border-primary-600'
                                : 'text-gray-500 border-transparent hover:text-gray-700'
                            }`}
                          onClick={() => setActiveTab(tab.id)}
                        >
                          <Icon className="w-4 h-4" />
                          {tab.label}
                          {count > 0 && (
                            <span
                              className={`ml-0.5 px-1.5 py-0.5 rounded-full text-[10px] font-mono ${
                                tab.urgent
                                  ? 'bg-primary-500/20 text-primary-700'
                                  : 'bg-gray-200 text-gray-600'
                              }`}
                            >
                              {count}
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            ))}
          </div>
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
                <div className="bg-gray-100 border border-gray-200 rounded-lg p-3">
                  <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div className="flex flex-wrap items-center gap-2">
                      <Button size="sm" variant="ghost" onClick={selectVisibleQueueItems}>
                        Select Visible
                      </Button>
                      <Button size="sm" variant="ghost" onClick={clearQueueSelection}>
                        Clear Selection
                      </Button>
                      <span className="text-xs text-gray-600">
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
                      <div className="text-xs text-gray-600">
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
                            <span className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">
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
                                  : 'bg-gray-100 text-gray-600 border border-gray-200'
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
          <Suspense fallback={<div className="p-6 text-sm text-gray-500">Loading…</div>}>
            <DecisionTraceTab
              decisionTraceAnalyticsData={decisionTraceAnalyticsData}
              decisionTraceAnalyticsLoading={decisionTraceAnalyticsLoading}
              decisionTraceData={decisionTraceData}
              decisionTraceLoading={decisionTraceLoading}
              refetchDecisionTrace={refetchDecisionTrace}
              refetchDecisionTraceAnalytics={refetchDecisionTraceAnalytics}
              traceViewsData={traceViewsData}
              setActiveTab={setActiveTab}
              applyTraceView={applyTraceView}
              buildAutonomousAgentsUrl={buildAutonomousAgentsUrl}
              buildTraceShareUrl={buildTraceShareUrl}
              collaborationUsers={collaborationUsers}
              currentTraceViewFilters={currentTraceViewFilters}
              decisionTraceActionMutation={decisionTraceActionMutation}
              expandedTraceEventId={expandedTraceEventId}
              setExpandedTraceEventId={setExpandedTraceEventId}
              location={location}
              navigate={navigate}
              queryClient={queryClient}
              selectedTraceViewId={selectedTraceViewId}
              setSelectedTraceViewId={setSelectedTraceViewId}
              traceActionNoteDrafts={traceActionNoteDrafts}
              setTraceActionNoteDrafts={setTraceActionNoteDrafts}
              traceActionableOnly={traceActionableOnly}
              setTraceActionableOnly={setTraceActionableOnly}
              traceActorModeFilter={traceActorModeFilter}
              setTraceActorModeFilter={setTraceActorModeFilter}
              traceAssignedToUserIdFilter={traceAssignedToUserIdFilter}
              setTraceAssignedToUserIdFilter={setTraceAssignedToUserIdFilter}
              traceAssigneeDrafts={traceAssigneeDrafts}
              setTraceAssigneeDrafts={setTraceAssigneeDrafts}
              traceCustomerFilter={traceCustomerFilter}
              setTraceCustomerFilter={setTraceCustomerFilter}
              traceDateRange={traceDateRange}
              setTraceDateRange={setTraceDateRange}
              traceDecisionTypeFilter={traceDecisionTypeFilter}
              setTraceDecisionTypeFilter={setTraceDecisionTypeFilter}
              traceDueAtDrafts={traceDueAtDrafts}
              setTraceDueAtDrafts={setTraceDueAtDrafts}
              traceEscalationStateFilter={traceEscalationStateFilter}
              setTraceEscalationStateFilter={setTraceEscalationStateFilter}
              traceFiltersDirtyRef={traceFiltersDirtyRef}
              traceOffset={traceOffset}
              setTraceOffset={setTraceOffset}
              traceOperatorPreset={traceOperatorPreset}
              setTraceOperatorPreset={setTraceOperatorPreset}
              tracePinnedOnly={tracePinnedOnly}
              setTracePinnedOnly={setTracePinnedOnly}
              traceSeverityFilter={traceSeverityFilter}
              setTraceSeverityFilter={setTraceSeverityFilter}
              traceSourceKindFilter={traceSourceKindFilter}
              setTraceSourceKindFilter={setTraceSourceKindFilter}
              traceStartAt={traceStartAt}
              traceStatusFilter={traceStatusFilter}
              setTraceStatusFilter={setTraceStatusFilter}
              traceTriageStatusFilter={traceTriageStatusFilter}
              setTraceTriageStatusFilter={setTraceTriageStatusFilter}
              traceUnassignedOnly={traceUnassignedOnly}
              setTraceUnassignedOnly={setTraceUnassignedOnly}
              traceViewIsDefaultDraft={traceViewIsDefaultDraft}
              setTraceViewIsDefaultDraft={setTraceViewIsDefaultDraft}
              traceViewNameDraft={traceViewNameDraft}
              setTraceViewNameDraft={setTraceViewNameDraft}
              userLabelById={userLabelById}
            />
          </Suspense>
        )}

        {activeTab === 'health' && (
          <Suspense fallback={<div className="p-6 text-sm text-gray-500">Loading…</div>}>
            <AutonomyHealthTab
              onDrillIntoInbox={openInboxHealthDrilldown}
              onDrillIntoQueue={openQueueHealthDrilldown}
              onOpenInboxForMonitorSignal={openInboxForMonitorSignal}
              healthPolicyDrafts={healthPolicyDrafts}
              setHealthPolicyDrafts={setHealthPolicyDrafts}
              location={location}
              monitorAnalyticsLoading={monitorAnalyticsLoading}
              refetchMonitorAnalytics={refetchMonitorAnalytics}
              setHealthAutonomyFilter={setHealthAutonomyFilter}
              setHealthBucketFilter={setHealthBucketFilter}
              setHealthMonitorTypeFilter={setHealthMonitorTypeFilter}
              healthPolicyEvaluations={healthPolicyEvaluations}
              setHealthPolicyEvaluations={setHealthPolicyEvaluations}
              healthPolicySimulations={healthPolicySimulations}
              setHealthPolicySimulations={setHealthPolicySimulations}
              healthCustomerRebalanceEvaluations={healthCustomerRebalanceEvaluations}
              setHealthCustomerRebalanceEvaluations={setHealthCustomerRebalanceEvaluations}
              formatAutonomyLabel={formatAutonomyLabel}
              formatReviewModeLabel={formatReviewModeLabel}
              healthAutonomyFilter={healthAutonomyFilter}
              healthBucketFilter={healthBucketFilter}
              healthMonitorTypeFilter={healthMonitorTypeFilter}
              loadPolicyEvaluationMutation={loadPolicyEvaluationMutation}
              rollbackMonitorPolicyMutation={rollbackMonitorPolicyMutation}
              updateMonitorPolicyMutation={updateMonitorPolicyMutation}
              monitorAnalyticsData={monitorAnalyticsData}
              buildAutonomousAgentsUrl={buildAutonomousAgentsUrl}
              deepLinkedHealthMonitor={deepLinkedHealthMonitor}
              filteredMonitorAnalytics={filteredMonitorAnalytics}
              healthCustomerFilter={healthCustomerFilter}
              healthCustomers={healthCustomers}
              navigate={navigate}
              openHealthPolicyComparison={openHealthPolicyComparison}
              setActiveTab={setActiveTab}
              setHealthCustomerFilter={setHealthCustomerFilter}
              setInboxCustomerFilter={setInboxCustomerFilter}
              setInboxHealthDrilldown={setInboxHealthDrilldown}
              setInboxJobFilter={setInboxJobFilter}
              setInboxPolicyDrilldown={setInboxPolicyDrilldown}
              setInboxSearch={setInboxSearch}
              setInboxStatusFilter={setInboxStatusFilter}
              setInboxTypeFilter={setInboxTypeFilter}
              setQueueCustomerFilter={setQueueCustomerFilter}
              setQueueHealthDrilldown={setQueueHealthDrilldown}
              setQueueJobFilter={setQueueJobFilter}
              setShowMonitorProfilesModal={setShowMonitorProfilesModal}
            />
          </Suspense>
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
                                <span className="text-xs px-2 py-0.5 rounded bg-gray-200 text-gray-700">{portfolio.status}</span>
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
                                <span className="text-xs px-2 py-0.5 rounded bg-gray-200 text-gray-700">{profile.status}</span>
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
          <Suspense fallback={<div className="p-6 text-sm text-gray-500">Loading…</div>}>
            <SwarmReviewTab
              swarmReviewJobs={swarmReviewJobs}
              swarmReviewJobsLoading={swarmReviewJobsLoading}
              refetchSwarmReviewJobs={refetchSwarmReviewJobs}
              swarmAnalyticsData={swarmAnalyticsData}
              swarmAnalyticsLoading={swarmAnalyticsLoading}
              refetchSwarmAnalytics={refetchSwarmAnalytics}
              visibilityScope={swarmReviewVisibilityScope}
              onVisibilityScopeChange={setSwarmReviewVisibilityScope}
              backlogBySwarmJobId={backlogBySwarmJobId}
              userLabelById={userLabelById}
              collaborationUsers={collaborationUsers}
              currentUserId={user?.id ? String(user.id) : undefined}
              noteDrafts={swarmReviewNoteDrafts}
              onNoteDraftsChange={setSwarmReviewNoteDrafts}
              actionMutation={actionMutation}
              createCodingBacklogMutation={createCodingBacklogMutation}
              onOpenJob={(job) => { setSelectedJob(job); setActiveTab('jobs'); }}
              onGoToBacklog={() => setActiveTab('backlog')}
            />
          </Suspense>
        )}

        {activeTab === 'outcomes' && (
          <SwarmOutcomesPanel
            outcomes={swarmOutcomes}
            userLabelById={userLabelById}
            onOpenJob={(jobId) => {
              setSelectedJob(null);
              navigate(buildAutonomousAgentsUrl(jobId));
              setActiveTab('jobs');
            }}
            onOpenBacklog={() => setActiveTab('backlog')}
          />
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
                                <span className={`text-xs px-2 py-1 rounded ${String(profile.status || '').toLowerCase() === 'active' ? 'bg-emerald-50 text-emerald-700 border border-emerald-100' : 'bg-gray-200 text-gray-700 border border-gray-200'}`}>
                                  {String(profile.status || 'active')}
                                </span>
                                <span className={`text-xs px-2 py-1 rounded ${String(profile.visibility || 'private').toLowerCase() === 'shared' ? 'bg-cyan-50 text-cyan-700 border border-cyan-100' : 'bg-gray-200 text-gray-700 border border-gray-200'}`}>
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
          <Suspense fallback={<div className="p-6 text-sm text-gray-500">Loading…</div>}>
            <CodingBacklogTab
              codingBacklogData={codingBacklogData}
              codingBacklogLoading={codingBacklogLoading}
              refetchCodingBacklog={refetchCodingBacklog}
              setActiveTab={setActiveTab}
              setSelectedJob={setSelectedJob}
              swarmOutcomeBySwarmJobId={swarmOutcomeBySwarmJobId}
              user={user}
              backlogAssignmentFilter={backlogAssignmentFilter}
              setBacklogAssignmentFilter={setBacklogAssignmentFilter}
              backlogCloseReasonDrafts={backlogCloseReasonDrafts}
              setBacklogCloseReasonDrafts={setBacklogCloseReasonDrafts}
              backlogCommandsText={backlogCommandsText}
              setBacklogCommandsText={setBacklogCommandsText}
              backlogFailureSymptom={backlogFailureSymptom}
              setBacklogFailureSymptom={setBacklogFailureSymptom}
              backlogFilePathsText={backlogFilePathsText}
              setBacklogFilePathsText={setBacklogFilePathsText}
              backlogGoal={backlogGoal}
              setBacklogGoal={setBacklogGoal}
              backlogItems={backlogItems}
              backlogNoteDrafts={backlogNoteDrafts}
              setBacklogNoteDrafts={setBacklogNoteDrafts}
              backlogQueueStateFilter={backlogQueueStateFilter}
              setBacklogQueueStateFilter={setBacklogQueueStateFilter}
              backlogSourceId={backlogSourceId}
              setBacklogSourceId={setBacklogSourceId}
              backlogTitle={backlogTitle}
              setBacklogTitle={setBacklogTitle}
              backlogVisibilityScope={backlogVisibilityScope}
              setBacklogVisibilityScope={setBacklogVisibilityScope}
              buildAutonomousAgentsUrl={buildAutonomousAgentsUrl}
              codeSources={codeSources}
              collaborationUsers={collaborationUsers}
              createCodingBacklogMutation={createCodingBacklogMutation}
              navigate={navigate}
              queryClient={queryClient}
              userLabelById={userLabelById}
            />
          </Suspense>
        )}

        <div className={activeTab === 'jobs' ? 'flex gap-4 flex-1 min-h-0' : 'hidden'}>
            {/* Jobs list */}
            <div className="w-2/3 flex flex-col">
              {/* Filters, collapsed by default.
                  Twelve controls in one row, and most visits to this page are
                  "what is running" rather than "narrow this down". The toggle
                  states how many are active, so a hidden filter can never
                  shape the list without saying so. */}
              <div className="flex items-center gap-2 mb-3">
                <button
                  type="button"
                  aria-expanded={showJobFilters}
                  onClick={() => setShowJobFilters((open) => !open)}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-sm rounded-lg border transition-colors duration-fast ${
                    activeJobFilterCount > 0
                      ? 'border-primary-500/60 bg-primary-500/10 text-primary-700'
                      : 'border-gray-300 text-gray-600 hover:border-gray-400'
                  }`}
                >
                  <Filter className="w-4 h-4" />
                  Filters
                  {activeJobFilterCount > 0 && (
                    <span className="px-1.5 py-0.5 rounded-full bg-primary-500/20 text-[11px] font-medium">
                      {activeJobFilterCount}
                    </span>
                  )}
                  {showJobFilters ? (
                    <ChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5" />
                  )}
                </button>
                {activeJobFilterCount > 0 && !showJobFilters && (
                  // Reachable without expanding: the most likely thing you want
                  // from a filter you did not mean to leave on is to remove it.
                  <button
                    type="button"
                    className="text-xs text-gray-500 hover:text-gray-900 underline"
                    onClick={() => {
                      setStatusFilter('');
                      setTypeFilter('');
                      setLaunchModeFilter('');
                      setHasRelaunchChildrenFilter('');
                      setRelaunchFromJobIdFilter('');
                      setGraphHealthFilter('');
                      setDedupSkipFilter('');
                      setScopeGuardFilter('');
                      setExperimentRecoveryFilter('');
                      setSwarmOnlyFilter(false);
                      setGraphSortBy('none');
                    }}
                  >
                    Clear
                  </button>
                )}
              </div>
              <div className={showJobFilters ? 'flex gap-3 mb-4 flex-wrap' : 'hidden'}>
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
                {jobCountSummary.quick_start_claude_backend > 0 && (
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
                )}
                {jobCountSummary.quick_start_domain_research > 0 && (
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
                )}
                {jobCountSummary.quick_start_bug_triage_swarm > 0 && (
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
                )}
                {jobCountSummary.quick_start_build_break_swarm > 0 && (
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
                )}
                {jobCountSummary.quick_start_frontend_regression_swarm > 0 && (
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
                )}
                {jobCountSummary.quick_start_repo_bug_triage > 0 && (
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
                )}
                {jobCountSummary.quick_start_role_workflow > 0 && (
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
                )}
                {jobCountSummary.critical > 0 && (
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
                )}
                {jobCountSummary.warning > 0 && (
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
                )}
                {jobCountSummary.ok > 0 && (
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
                )}
                {jobCountSummary.unknown > 0 && (
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
                )}
                {jobCountSummary.dedup_gt0 > 0 && (
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
                )}
                {jobCountSummary.dedup_gte3 > 0 && (
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
                )}
                {jobCountSummary.scope_guard_blocked > 0 && (
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
                )}
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
                {jobCountSummary.bootstrap_recovered > 0 && (
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
                )}
                {jobCountSummary.fallback_attempted > 0 && (
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
                )}
                {jobCountSummary.unresolved_recovery_jobs > 0 && (
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
                )}
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
                <SkeletonList rows={5} label="Loading jobs" className="flex-1" />
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

            {/* Detail panel. Collapsed, it is a rail rather than nothing: the
                control to bring it back has to live somewhere the user can
                see, and inside the panel that just closed is not that. */}
            {detailPanelCollapsed ? (
              <div className="w-10 shrink-0 flex flex-col items-center pt-2 border-l border-gray-200">
                <button
                  type="button"
                  aria-label="Show detail panel"
                  title="Show detail panel"
                  className="p-1.5 rounded-md text-gray-500 hover:bg-gray-200 hover:text-gray-900 transition-all duration-fast ease-ui"
                  onClick={() => setDetailPanelCollapsed(false)}
                >
                  <PanelRightOpen className="w-4 h-4" />
                </button>
                {selectedJob && (
                  <span
                    className="mt-2 live-dot"
                    title={`${selectedJob.name} is selected`}
                    aria-hidden="true"
                  />
                )}
              </div>
            ) : (
            <div className="w-1/3 shrink-0 relative">
              <button
                type="button"
                aria-label="Collapse detail panel"
                title="Collapse detail panel"
                className="absolute right-1 top-1 z-10 p-1.5 rounded-md text-gray-500 hover:bg-gray-200 hover:text-gray-900 transition-all duration-fast ease-ui"
                onClick={() => setDetailPanelCollapsed(true)}
              >
                <PanelRightClose className="w-4 h-4" />
              </button>
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
            )}
        </div>

        {activeTab === 'templates' && (
          <Suspense fallback={<div className="p-6 text-sm text-gray-500">Loading…</div>}>
            <JobTemplatesTab
              templates={templatesData?.templates || []}
              quickStarts={templateQuickStarts}
              claudeBackendAvailable={Boolean(claudeBackendTemplate)}
              scope={templateRecommendScope}
              onScopeChange={setTemplateRecommendScope}
              goal={templateRecommendGoal}
              onGoalChange={setTemplateRecommendGoal}
              onSelectTemplate={setCreateFromTemplate}
            />
          </Suspense>
        )}

        {activeTab === 'chains' && (
          <Suspense fallback={<div className="p-6 text-sm text-gray-500">Loading…</div>}>
            <JobChainsTab
              chains={displayedChainDefinitions}
              onStartChain={setStartFromChain}
            />
          </Suspense>
        )}

        {activeTab === 'inbox' && (
          <Suspense fallback={<div className="p-6 text-sm text-gray-500">Loading…</div>}>
            <ResearchInboxTab
              chainsData={chainsData}
              inboxLoading={inboxLoading}
              inboxStats={inboxStats}
              myPreferences={myPreferences}
              refetchInbox={refetchInbox}
              setActiveTab={setActiveTab}
              setShowInboxMonitorModal={setShowInboxMonitorModal}
              setShowMonitorProfilesModal={setShowMonitorProfilesModal}
              activeFollowUpReviewKey={activeFollowUpReviewKey}
              buildAutonomousAgentsUrl={buildAutonomousAgentsUrl}
              createFromChainMutation={createFromChainMutation}
              createMutation={createMutation}
              followUpQueueActionMutation={followUpQueueActionMutation}
              followUpReviewNoteDrafts={followUpReviewNoteDrafts}
              setFollowUpReviewNoteDrafts={setFollowUpReviewNoteDrafts}
              healthCustomers={healthCustomers}
              inboxBulkFollowUpNote={inboxBulkFollowUpNote}
              setInboxBulkFollowUpNote={setInboxBulkFollowUpNote}
              inboxBulkRejectReason={inboxBulkRejectReason}
              setInboxBulkRejectReason={setInboxBulkRejectReason}
              inboxCustomerFilter={inboxCustomerFilter}
              setInboxCustomerFilter={setInboxCustomerFilter}
              inboxHealthDrilldown={inboxHealthDrilldown}
              setInboxHealthDrilldown={setInboxHealthDrilldown}
              inboxJobFilter={inboxJobFilter}
              setInboxJobFilter={setInboxJobFilter}
              inboxPolicyDrilldown={inboxPolicyDrilldown}
              setInboxPolicyDrilldown={setInboxPolicyDrilldown}
              inboxRejectReasonDrafts={inboxRejectReasonDrafts}
              setInboxRejectReasonDrafts={setInboxRejectReasonDrafts}
              inboxSearch={inboxSearch}
              setInboxSearch={setInboxSearch}
              inboxStatusFilter={inboxStatusFilter}
              setInboxStatusFilter={setInboxStatusFilter}
              inboxTypeFilter={inboxTypeFilter}
              setInboxTypeFilter={setInboxTypeFilter}
              location={location}
              navigate={navigate}
              paperRepoSelectionDrafts={paperRepoSelectionDrafts}
              setPaperRepoSelectionDrafts={setPaperRepoSelectionDrafts}
              queryClient={queryClient}
              selectedInboxIds={selectedInboxIds}
              setSelectedInboxIds={setSelectedInboxIds}
              selectedInboxItems={selectedInboxItems}
              unsafeExecBadge={unsafeExecBadge}
              upsertMonitorProfileMutation={upsertMonitorProfileMutation}
              visibleInboxItems={visibleInboxItems}
            />
          </Suspense>
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
      {showNewCampaignModal && (
        <NewCampaignModal
          onClose={() => setShowNewCampaignModal(false)}
          onCreated={() => {
            queryClient.invalidateQueries(['agent-jobs']);
            queryClient.invalidateQueries(['agent-jobs-stats']);
          }}
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
