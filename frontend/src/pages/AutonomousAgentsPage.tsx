/**
 * Autonomous Agents Page
 *
 * Manage and monitor autonomous agent jobs that run independently
 * to accomplish goals like research, monitoring, and analysis.
 */

import React, { Suspense, lazy, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
  formatAutonomyLabel,
  formatReviewModeLabel,
} from '../components/agent/autonomyShared';
import type {
} from '../components/agent/autonomyShared';

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
  Layers,
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
  CodingBacklogItem,
  CodingBacklogItemCreate,
  CodingSwarmProfile,
  CodingSwarmProfileCreate,
  CodingSwarmProfileUpdate,
  AgentJobTemplate,
  AgentJobStatus,
  AgentJobChainDefinition,
  AgentJobChainStatus,
  ResearchMonitorAnalyticsResponse,
  ResearchMonitorCustomerRebalanceEvaluationDetail,
  ResearchMonitorPolicyEvaluationDetail,
  ResearchMonitorPolicySimulationResponse,
} from '../types';
import {
} from '../utils/agentMemoryExtraction';
import { mergeProgressUpdateIntoJob, TERMINAL_JOB_STATUSES } from '../utils/agentJobProgress';
import {
  getExperimentRecoveryPriority as getExperimentRecoveryPriorityForRun,
  summarizeExperimentRun,
} from '../utils/experimentRunSummary';
import Button from '../components/common/Button';
import {
  buildResearchInboxUrl,
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
} from '../utils/agentJobDetail';
import { swarmQuickStartPreset } from '../components/agent/swarmQuickStarts';
import {
  buildBugTriageSwarmQuickStartPayload,
  buildBuildBreakSwarmQuickStartPayload,
  buildFrontendRegressionSwarmQuickStartPayload,
} from './autonomousAgentQuickStarts';
import {
  useCreateAgentJobMutation,
  useCreateJobFromChainMutation,
  useFollowUpQueueActionMutation,
  useUpsertMonitorProfileMutation,
} from '../components/agent/agentJobMutations';


// Tabs load when their tab is opened, not when the page is.
//
// Extraction alone did not shrink the bundle -- the components landed in the
// same chunk, so opening "My Jobs" still downloaded Swarm Review. Splitting
// is what extraction was *for*; it just is not what extraction *is*.
const JobChainsTab = lazy(() => import('../components/agent/tabs/JobChainsTab'));
const SwarmReviewTab = lazy(() => import('../components/agent/tabs/SwarmReviewTab'));
const DecisionTraceTab = lazy(() => import('../components/agent/tabs/DecisionTraceTab'));
const OperatorQueueTab = lazy(() => import('../components/agent/tabs/OperatorQueueTab'));
const SwarmProfilesTab = lazy(() => import('../components/agent/tabs/SwarmProfilesTab'));
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
    ],
  },
  {
    name: 'Setup',
    tabs: [
      { id: 'profiles', label: 'Swarm Profiles', icon: Settings },
    ],
  },
];


const AutonomousAgentsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AgentJobsTab>('jobs');
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
  // Stays on the page: it is a query key. Both the swarm-review job list
  // and the analytics refetch when it changes, and those queries live here.
  const [swarmReviewVisibilityScope, setSwarmReviewVisibilityScope] =
    useState<'mine' | 'shared' | 'all'>('mine');
  const [swarmReviewNoteDrafts, setSwarmReviewNoteDrafts] = useState<Record<string, string>>({});
  const [exportingJob, setExportingJob] = useState<AgentJob | null>(null);
  const landingTabInitializedRef = useRef(false);

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

  const queryClient = useQueryClient();
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();
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
  const deepLinkedHealthCustomer = useMemo(() => new URLSearchParams(location.search).get('health_customer'), [location.search]);
  const deepLinkedHealthMonitor = useMemo(() => new URLSearchParams(location.search).get('health_monitor'), [location.search]);
  const deepLinkedHealthPolicyHistory = useMemo(() => new URLSearchParams(location.search).get('health_policy_history'), [location.search]);
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






  // The inbox is its own page now, so a drilldown is a navigation and the URL
  // is the whole of the state being handed over.
  const openInboxHealthDrilldown = useCallback((
    drilldown: InboxHealthDrilldown,
    context?: { customer?: string | null; monitorJobId?: string | null }
  ) => {
    navigate(buildResearchInboxUrl({
      inbox_status: 'accepted',
      inbox_customer: String(context?.customer || '').trim() || null,
      inbox_job: String(context?.monitorJobId || '').trim() || null,
      inbox_health_drilldown: drilldown || null,
    }));
  }, [navigate]);

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
    // This effect's whole job is interpreting *this page's* URL. Once a link
    // here leaves for another page -- the inbox moved to /research/inbox -- the
    // parameters it reads are someone else's, and acting on them made the page
    // renavigate against a router that had already moved on. Measured as an
    // unbounded render loop from a single "View Inbox" click.
    if (!location.pathname.startsWith('/autonomous-agents')) return;
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
    if (deepLinkedDomainTab || deepLinkedFleetTab) {
      // Both moved out of Runs: portfolios to the R&D door, profiles to
      // Settings. Existing links keep working, carrying what they pointed at.
      const p = new URLSearchParams(location.search);
      const carry = new URLSearchParams();
      ['fleetId', 'profileId', 'opportunityId'].forEach((k) => {
        const v = String(p.get(k) || '').trim();
        if (v) carry.set(k, v);
      });
      const q = carry.toString();
      navigate(
        (deepLinkedFleetTab ? '/research/fleet' : '/settings/domain-profiles') + (q ? `?${q}` : ''),
        { replace: true }
      );
      return;
    }
    if (deepLinkedInboxTab) {
      // The inbox moved to the Library. Old links -- ?tab=inbox, with whatever
      // filters they carried -- still land on it rather than on a tab that no
      // longer exists.
      const inboxParams = new URLSearchParams(location.search);
      navigate(buildResearchInboxUrl({
        inbox: inboxParams.get('inbox'),
        inbox_job: inboxParams.get('inbox_job'),
        inbox_customer: inboxParams.get('inbox_customer'),
        inbox_health_drilldown: inboxParams.get('inbox_health_drilldown'),
        inbox_policy_drilldown: inboxParams.get('inbox_policy_drilldown'),
      }), { replace: true });
      return;
    }
    const normalizedQueueCustomer = String(deepLinkedQueueCustomer || '').trim();
    const normalizedQueueJobId = String(deepLinkedQueueJobId || '').trim();
    const normalizedHealthCustomer = String(deepLinkedHealthCustomer || '').trim();
    if (normalizedQueueCustomer !== queueCustomerFilter) {
      setQueueCustomerFilter(normalizedQueueCustomer);
    }
    if (normalizedQueueJobId !== queueJobFilter) {
      setQueueJobFilter(normalizedQueueJobId);
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
  }, [deepLinkedTraceTab, deepLinkedHealthTab, deepLinkedJobId, deepLinkedJobData, deepLinkedJobError, deepLinkedQueueTab, deepLinkedQueueCustomer, deepLinkedQueueJobId, deepLinkedQueueHealthDrilldown, deepLinkedDomainTab, deepLinkedFleetTab, deepLinkedInboxTab, deepLinkedHealthCustomer, healthCustomerFilter, queueCustomerFilter, queueHealthDrilldown, queueJobFilter, jobsData, navigate, buildAutonomousAgentsUrl, location.search, location.pathname]);



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
      enabled: activeTab === 'health',
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
  const { data: codingBacklogData } = useQuery(
    // Only the swarm and outcomes tabs read this now: they derive
    // backlogBySwarmJobId and want every item, not one person's filters.
    ['coding-backlog-items', 'all', ''],
    () => apiClient.listCodingBacklogItems({
      limit: 100,
      offset: 0,
      visibility_scope: 'all',
    }),
    {
      enabled: activeTab === 'swarm' || activeTab === 'outcomes',
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



  // Research Inbox


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
      navigate(buildResearchInboxUrl({
        inbox_status: 'accepted',
        inbox_job: String(monitorJobId || '').trim() || null,
        inbox: String(inboxItemId || '').trim() || null,
        inbox_policy_drilldown: String(policyDrilldown || '').trim() || null,
      }));
    },
    [navigate]
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



  const upsertMonitorProfileMutation = useUpsertMonitorProfileMutation();










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






  const createMutation = useCreateAgentJobMutation({
    onCreated: (job) => {
      setShowCreateModal(false);
      setActiveTab('jobs');
      setSelectedJob(job);
    },
  });

  const createCodingBacklogMutation = useMutation(
    (data: CodingBacklogItemCreate) => apiClient.createCodingBacklogItem(data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['coding-backlog-items']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Coding backlog item created');
        // The form those resets cleared lives on /coding-backlog now. Here the
        // callers are the swarm review tab and the job detail panel, which
        // create an item from something already on screen and have no form.
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

  const createFromChainMutation = useCreateJobFromChainMutation({
    onCreated: (job) => {
      setStartFromChain(null);
      setActiveTab('jobs');
      setSelectedJob(job);
    },
  });


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





  const openQueueItemTarget = useCallback((item: AgentCheckpointQueueItem) => {
    // A queue item can point at an opportunity on either surface; both are
    // their own pages now, so opening one is a navigation carrying the ids the
    // page focuses on.
    const focusUrl = (base: string, ownerKey: string, ownerId: string, opportunityId?: string | null) => {
      const carry = new URLSearchParams({ [ownerKey]: ownerId });
      const opp = String(opportunityId || '').trim();
      if (opp) carry.set('opportunityId', opp);
      return `${base}?${carry.toString()}`;
    };
    if (item.domain_research_profile_id) {
      navigate(focusUrl('/settings/domain-profiles', 'profileId',
        String(item.domain_research_profile_id), item.profile_opportunity_id));
      return;
    }
    if (item.portfolio_id) {
      navigate(focusUrl('/research/fleet', 'fleetId',
        String(item.portfolio_id), item.portfolio_opportunity_id));
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



  const followUpQueueActionMutation = useFollowUpQueueActionMutation({
    // Domain profiles and portfolios are their own pages now; Runs has no
    // list of either to refresh after an action.
    onFollowUpLaunched: (followUpJobId) => {
      setActiveTab('jobs');
      navigate(buildAutonomousAgentsUrl(followUpJobId), { replace: true });
      return true;
    },
  });
















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
          <Suspense fallback={<div className="p-6 text-sm text-gray-500">Loading…</div>}>
            <OperatorQueueTab
              checkpointQueueData={checkpointQueueData}
              checkpointQueueLoading={checkpointQueueLoading}
              refetchCheckpointQueue={refetchCheckpointQueue}
              setActiveTab={setActiveTab}
              setHealthCustomerFilter={setHealthCustomerFilter}
              setQueueDrafts={setQueueDrafts}
              setShowInboxMonitorModal={setShowInboxMonitorModal}
              actionMutation={actionMutation}
              buildAutonomousAgentsUrl={buildAutonomousAgentsUrl}
              createFromChainMutation={createFromChainMutation}
              createMutation={createMutation}
              followUpQueueActionMutation={followUpQueueActionMutation}
              getQueueDraft={getQueueDraft}
              getQueueDraftValue={getQueueDraftValue}
              navigate={navigate}
              openHealthPolicyComparison={openHealthPolicyComparison}
              openQueueItemTarget={openQueueItemTarget}
              queryClient={queryClient}
              queueBulkNote={queueBulkNote}
              setQueueBulkNote={setQueueBulkNote}
              queueCustomerFilter={queueCustomerFilter}
              setQueueCustomerFilter={setQueueCustomerFilter}
              queueEscalationFilter={queueEscalationFilter}
              setQueueEscalationFilter={setQueueEscalationFilter}
              queueHealthDrilldown={queueHealthDrilldown}
              setQueueHealthDrilldown={setQueueHealthDrilldown}
              queueItemTypeFilter={queueItemTypeFilter}
              setQueueItemTypeFilter={setQueueItemTypeFilter}
              queueJobFilter={queueJobFilter}
              setQueueJobFilter={setQueueJobFilter}
              queueJobTypeFilter={queueJobTypeFilter}
              setQueueJobTypeFilter={setQueueJobTypeFilter}
              queueOperatorPreset={queueOperatorPreset}
              setQueueOperatorPreset={setQueueOperatorPreset}
              queueOverdueOnly={queueOverdueOnly}
              setQueueOverdueOnly={setQueueOverdueOnly}
              queueSelection={queueSelection}
              setQueueSelection={setQueueSelection}
              queueSlaBucketFilter={queueSlaBucketFilter}
              setQueueSlaBucketFilter={setQueueSlaBucketFilter}
              queueSortBy={queueSortBy}
              setQueueSortBy={setQueueSortBy}
              queueStatusFilter={queueStatusFilter}
              setQueueStatusFilter={setQueueStatusFilter}
              rollbackMonitorPolicyMutation={rollbackMonitorPolicyMutation}
              selectedQueueItems={selectedQueueItems}
              updateMonitorPolicyMutation={updateMonitorPolicyMutation}
              visibleQueueItems={visibleQueueItems}
            />
          </Suspense>
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
              setQueueCustomerFilter={setQueueCustomerFilter}
              setQueueHealthDrilldown={setQueueHealthDrilldown}
              setQueueJobFilter={setQueueJobFilter}
              setShowMonitorProfilesModal={setShowMonitorProfilesModal}
            />
          </Suspense>
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
              onGoToBacklog={() => navigate('/coding-backlog')}
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
            onOpenBacklog={() => navigate('/coding-backlog')}
          />
        )}

        {activeTab === 'profiles' && (
          <Suspense fallback={<div className="p-6 text-sm text-gray-500">Loading…</div>}>
            <SwarmProfilesTab
              setActiveTab={setActiveTab}
              setCodingSwarmLaunchSeed={setCodingSwarmLaunchSeed}
              setShowBugTriageSwarmQuickStartModal={setShowBugTriageSwarmQuickStartModal}
              setShowBuildBreakSwarmQuickStartModal={setShowBuildBreakSwarmQuickStartModal}
              setShowFrontendRegressionSwarmQuickStartModal={setShowFrontendRegressionSwarmQuickStartModal}
              user={user}
              codeSources={codeSources}
              codingSwarmProfileDraft={codingSwarmProfileDraft}
              setCodingSwarmProfileDraft={setCodingSwarmProfileDraft}
              codingSwarmProfiles={codingSwarmProfiles}
              collaborationUsers={collaborationUsers}
              createCodingSwarmProfileMutation={createCodingSwarmProfileMutation}
              deleteCodingSwarmProfileMutation={deleteCodingSwarmProfileMutation}
              editingCodingSwarmProfileId={editingCodingSwarmProfileId}
              setEditingCodingSwarmProfileId={setEditingCodingSwarmProfileId}
              profileDefaultOnly={profileDefaultOnly}
              setProfileDefaultOnly={setProfileDefaultOnly}
              profileOwnerFilter={profileOwnerFilter}
              setProfileOwnerFilter={setProfileOwnerFilter}
              profileOwnershipFilter={profileOwnershipFilter}
              setProfileOwnershipFilter={setProfileOwnershipFilter}
              profilePresetFilter={profilePresetFilter}
              setProfilePresetFilter={setProfilePresetFilter}
              profileSourceFilter={profileSourceFilter}
              setProfileSourceFilter={setProfileSourceFilter}
              profileStatusFilter={profileStatusFilter}
              setProfileStatusFilter={setProfileStatusFilter}
              profileVisibilityFilter={profileVisibilityFilter}
              setProfileVisibilityFilter={setProfileVisibilityFilter}
              queryClient={queryClient}
              updateCodingSwarmProfileMutation={updateCodingSwarmProfileMutation}
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
