import React, { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Activity, Bot, GitBranch, GitFork, Layers, MemoryStick, Route, Workflow } from 'lucide-react';
import toast from 'react-hot-toast';

import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { apiClient } from '../services/api';
import type {
  AgentControlRunDetail,
  AgentControlRunEdge,
  AgentControlRunNode,
  AgentControlRunReviewItem,
  AgentControlRunSummary,
  AgentControlRunView,
} from '../types';

type QueueHealthDrilldown = '' | 'pending_follow_up_approvals' | 'manual_follow_up_recommendations' | 'blocked_follow_up';
type QueuePreset = '' | 'approval_required' | 'failed_recovery' | 'compiler';
type QueueScope = 'selected_run' | 'global';
type QueueSort = 'priority' | 'created_at_desc' | 'age_desc';

const GLOBAL_QUEUE_PAGE_SIZE = 50;

const statusTone = (status?: string | null) => {
  switch ((status || '').toLowerCase()) {
    case 'completed':
      return 'bg-emerald-100 text-emerald-800';
    case 'failed':
      return 'bg-rose-100 text-rose-800';
    case 'running':
      return 'bg-sky-100 text-sky-800';
    case 'blocked':
      return 'bg-amber-100 text-amber-800';
    default:
      return 'bg-gray-200 text-gray-700';
  }
};

const kindIcon = (kind: string) => {
  switch (kind) {
    case 'agent_job':
      return <Bot className="h-4 w-4" />;
    case 'workflow_execution':
      return <Workflow className="h-4 w-4" />;
    case 'decision_event':
      return <Activity className="h-4 w-4" />;
    case 'experiment_plan':
    case 'experiment_run':
      return <GitBranch className="h-4 w-4" />;
    default:
      return <Layers className="h-4 w-4" />;
  }
};

const formatDateTime = (value?: string | null) => {
  if (!value) return '—';
  return new Date(value).toLocaleString();
};

const summarizeSchedulerState = (schedulerState?: Record<string, any> | null) => {
  if (!schedulerState || typeof schedulerState !== 'object') return [] as string[];
  const lines: string[] = [];
  const queueReason = String(schedulerState.queue_reason || '').trim();
  const lastError = String(schedulerState.last_error || '').trim();
  const retryCount = Number(schedulerState.retry_count ?? schedulerState.consecutive_failures ?? NaN);
  const backoffUntil = String(schedulerState.backoff_until || '').trim();
  if (queueReason) lines.push(`Reason: ${queueReason.replace(/_/g, ' ')}`);
  if (Number.isFinite(retryCount)) lines.push(`Retries: ${retryCount}`);
  if (backoffUntil) lines.push(`Backoff until: ${formatDateTime(backoffUntil)}`);
  if (lastError) lines.push(`Last error: ${lastError}`);
  return lines;
};

const buildCheckpointDraftKey = (review: {
  queue_item_key?: string | null;
  source_kind?: string | null;
  source_id?: string | null;
  opportunity_id?: string | null;
  review_type?: string | null;
}) => buildReviewSelectionKey(review);

const getInitialCheckpointDraft = (review: AgentControlRunReviewItem) => {
  const draft = review.checkpoint_action_draft && typeof review.checkpoint_action_draft === 'object'
    ? review.checkpoint_action_draft
    : review.checkpoint?.action && typeof review.checkpoint.action === 'object'
      ? review.checkpoint.action
      : {};
  const paramsValue = draft && typeof draft.params === 'object' && draft.params !== null ? draft.params : {};
  return {
    tool: String(draft?.tool || '').trim(),
    purpose: String(draft?.purpose || '').trim(),
    params: JSON.stringify(paramsValue, null, 2),
  };
};

const buildReviewQueueKey = (review: {
  source_kind?: string | null;
  source_id?: string | null;
  opportunity_id?: string | null;
  review_type?: string | null;
}) =>
  [
    String(review.source_kind || '').trim(),
    String(review.source_id || '').trim(),
    String(review.opportunity_id || '').trim(),
    String(review.review_type || '').trim(),
  ].join('::');

const buildReviewSelectionKey = (review: {
  queue_item_key?: string | null;
  source_kind?: string | null;
  source_id?: string | null;
  opportunity_id?: string | null;
  review_type?: string | null;
}) => String(review.queue_item_key || '').trim() || buildReviewQueueKey(review);

const normalizeReviewBulkClass = (review: {
  item_type?: string | null;
  review_type?: string | null;
  follow_up_launch_status?: string | null;
  source_kind?: string | null;
}) => {
  const itemType = String(review.item_type || review.review_type || '').trim();
  if (itemType === 'approval_checkpoint' || itemType === 'job_recovery') return itemType;
  if (
    itemType === 'follow_up_recommendation' &&
    String(review.follow_up_launch_status || '').trim() === 'pending_approval' &&
    (String(review.source_kind || '').trim() === 'profile' || String(review.source_kind || '').trim() === 'portfolio')
  ) {
    return 'follow_up_recommendation';
  }
  return '';
};

const resolveDecisionTraceDeepLink = (targetTab?: string | null, params?: Record<string, string> | null) => {
  const tab = String(targetTab || '').trim();
  const query = params ? `?${new URLSearchParams(params).toString()}` : '';
  if (tab === 'domain' || tab === 'fleet' || tab === 'jobs' || tab === 'job' || tab === 'trace' || tab === 'queue') {
    return `/autonomous-agents${query}`;
  }
  if (!tab) return `/autonomous-agents${query}`;
  return `/${tab}${query}`;
};

const buildHealthMonitorPath = (
  review: Pick<AgentControlRunReviewItem, 'customer' | 'job_id' | 'policy_guardrail_target_history_entry_id'>,
  options?: { includePolicyHistory?: boolean }
) => {
  const params = new URLSearchParams();
  params.set('tab', 'health');
  const customer = String(review.customer || '').trim();
  const monitorJobId = String(review.job_id || '').trim();
  const historyEntryId = String(review.policy_guardrail_target_history_entry_id || '').trim();
  if (customer) params.set('health_customer', customer);
  if (monitorJobId) params.set('health_monitor', monitorJobId);
  if (options?.includePolicyHistory && historyEntryId) params.set('health_policy_history', historyEntryId);
  return `/autonomous-agents?${params.toString()}`;
};

const CONTROL_VIEW_QUERY_KEYS = [
  'type',
  'outcome',
  'routingTier',
  'run',
  'hasOperatorReview',
  'reviewType',
  'reviewStatus',
  'queueStatus',
  'queueCustomer',
  'queueSla',
  'queueEscalation',
  'queueHealthDrilldown',
  'queuePreset',
  'queueScope',
  'queueSort',
] as const;

const normalizeQueueHealthDrilldown = (value?: string | null): QueueHealthDrilldown => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'pending_follow_up_approvals') return 'pending_follow_up_approvals';
  if (normalized === 'manual_follow_up_recommendations') return 'manual_follow_up_recommendations';
  if (normalized === 'blocked_follow_up') return 'blocked_follow_up';
  return '';
};

const normalizeQueuePreset = (value?: string | null): QueuePreset => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'approval_required') return 'approval_required';
  if (normalized === 'failed_recovery') return 'failed_recovery';
  if (normalized === 'compiler') return 'compiler';
  return '';
};

const normalizeQueueScope = (value?: string | null): QueueScope => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'global') return 'global';
  return 'selected_run';
};

const normalizeQueueSort = (value?: string | null): QueueSort => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'created_at_desc') return 'created_at_desc';
  if (normalized === 'age_desc') return 'age_desc';
  return 'priority';
};

const parseQueueOffset = (value?: string | null) => {
  const parsed = Number.parseInt(String(value || '').trim(), 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
};

const queueDrilldownLabel = (value: QueueHealthDrilldown) => {
  if (value === 'pending_follow_up_approvals') return 'pending approvals';
  if (value === 'manual_follow_up_recommendations') return 'manual recommendations';
  if (value === 'blocked_follow_up') return 'blocked follow-ups';
  return 'queue slice';
};

const queuePresetLabel = (value: QueuePreset) => {
  if (value === 'approval_required') return 'approval required';
  if (value === 'failed_recovery') return 'failed recovery';
  if (value === 'compiler') return 'compiler';
  return 'queue preset';
};

const includesCompilerToken = (...values: Array<string | null | undefined>) =>
  values.some((value) => String(value || '').trim().toLowerCase().includes('compiler'));

const reviewMatchesQueueFilters = ({
  review,
  reviewType,
  reviewStatus,
  queueStatus,
  queueCustomer,
  queueSla,
  queueEscalation,
  queueHealthDrilldown,
  queuePreset,
}: {
  review: AgentControlRunReviewItem;
  reviewType: string;
  reviewStatus: string;
  queueStatus: string;
  queueCustomer: string;
  queueSla: string;
  queueEscalation: string;
  queueHealthDrilldown: QueueHealthDrilldown;
  queuePreset: QueuePreset;
}) => {
  if (reviewType && String(review.review_type || '').trim().toLowerCase() !== reviewType.toLowerCase()) return false;
  if (reviewStatus && String(review.review_status || '').trim().toLowerCase() !== reviewStatus.toLowerCase()) return false;

  const statusTokens = [
    review.status,
    review.follow_up_launch_status,
    review.follow_up_review_status,
  ]
    .map((value) => String(value || '').trim().toLowerCase())
    .filter(Boolean);
  if (queueStatus && !statusTokens.includes(queueStatus.toLowerCase())) return false;
  if (queueCustomer && String(review.customer || '').trim().toLowerCase() !== queueCustomer.toLowerCase()) return false;
  if (queueSla && String(review.sla_bucket || '').trim().toLowerCase() !== queueSla.toLowerCase()) return false;
  if (queueEscalation && String(review.escalation_level || '').trim().toLowerCase() !== queueEscalation.toLowerCase()) return false;

  if (queueHealthDrilldown === 'pending_follow_up_approvals') {
    if (!(review.review_type === 'follow_up_recommendation' && String(review.follow_up_launch_status || '').trim().toLowerCase() === 'pending_approval')) {
      return false;
    }
  } else if (queueHealthDrilldown === 'manual_follow_up_recommendations') {
    if (review.review_type !== 'manual_follow_up_recommendation') return false;
  } else if (queueHealthDrilldown === 'blocked_follow_up') {
    const blocked =
      !!String(review.follow_up_block_reason || '').trim() ||
      review.review_type === 'budget_review' ||
      review.review_type === 'policy_review';
    if (!blocked) return false;
  }

  if (queuePreset === 'approval_required') {
    const isApprovalRequired =
      review.review_type === 'approval_checkpoint' ||
      (review.review_type === 'follow_up_recommendation' &&
        String(review.follow_up_launch_status || '').trim().toLowerCase() === 'pending_approval');
    if (!isApprovalRequired) return false;
  } else if (queuePreset === 'failed_recovery') {
    if (review.review_type !== 'job_recovery') return false;
  } else if (queuePreset === 'compiler') {
    if (
      !includesCompilerToken(
        review.customer,
        review.title,
        review.summary,
        review.evidence_summary,
        review.job_name,
        review.job_type
      )
    ) {
      return false;
    }
  }

  return true;
};

const buildCurrentControlViewFilters = (params: URLSearchParams) => {
  const filters: Record<string, string> = {};
  const sourceType = String(params.get('type') || '').trim();
  const outcome = String(params.get('outcome') || '').trim();
  const routingTier = String(params.get('routingTier') || '').trim();
  const selectedRunId = String(params.get('run') || '').trim();
  const hasOperatorReview = String(params.get('hasOperatorReview') || '').trim();
  const reviewType = String(params.get('reviewType') || '').trim();
  const reviewStatus = String(params.get('reviewStatus') || '').trim();
  const queueStatus = String(params.get('queueStatus') || '').trim();
  const queueCustomer = String(params.get('queueCustomer') || '').trim();
  const queueSla = String(params.get('queueSla') || '').trim();
  const queueEscalation = String(params.get('queueEscalation') || '').trim();
  const queueHealthDrilldown = normalizeQueueHealthDrilldown(params.get('queueHealthDrilldown'));
  const queuePreset = normalizeQueuePreset(params.get('queuePreset'));
  const queueScope = normalizeQueueScope(params.get('queueScope'));
  const queueSort = normalizeQueueSort(params.get('queueSort'));
  if (sourceType) filters.source_type = sourceType;
  if (outcome) filters.outcome = outcome;
  if (routingTier) filters.routing_tier = routingTier;
  if (selectedRunId) filters.selected_run_id = selectedRunId;
  if (hasOperatorReview) filters.has_operator_review = hasOperatorReview;
  if (reviewType) filters.review_type = reviewType;
  if (reviewStatus) filters.review_status = reviewStatus;
  if (queueStatus) filters.queue_status = queueStatus;
  if (queueCustomer) filters.queue_customer = queueCustomer;
  if (queueSla) filters.queue_sla = queueSla;
  if (queueEscalation) filters.queue_escalation = queueEscalation;
  if (queueHealthDrilldown) filters.queue_health_drilldown = queueHealthDrilldown;
  if (queuePreset) filters.queue_preset = queuePreset;
  if (queueScope !== 'selected_run') filters.queue_scope = queueScope;
  if (queueSort !== 'priority') filters.queue_sort = queueSort;
  return filters;
};

const applyControlViewToParams = ({
  params,
  view,
  preserveExplicit,
}: {
  params: URLSearchParams;
  view: AgentControlRunView;
  preserveExplicit: boolean;
}) => {
  const next = new URLSearchParams(params);
  const filters = view.filters || {};
  next.set('view', view.id);

  const mappings: Array<[typeof CONTROL_VIEW_QUERY_KEYS[number], string]> = [
    ['type', String(filters.source_type || '').trim()],
    ['outcome', String(filters.outcome || '').trim()],
    ['routingTier', String(filters.routing_tier || '').trim()],
    ['run', String(filters.selected_run_id || '').trim()],
    ['hasOperatorReview', String(filters.has_operator_review || '').trim()],
    ['reviewType', String(filters.review_type || '').trim()],
    ['reviewStatus', String(filters.review_status || '').trim()],
    ['queueStatus', String(filters.queue_status || '').trim()],
    ['queueCustomer', String(filters.queue_customer || '').trim()],
    ['queueSla', String(filters.queue_sla || '').trim()],
    ['queueEscalation', String(filters.queue_escalation || '').trim()],
    ['queueHealthDrilldown', normalizeQueueHealthDrilldown(filters.queue_health_drilldown)],
    ['queuePreset', normalizeQueuePreset(filters.queue_preset)],
    ['queueScope', normalizeQueueScope(filters.queue_scope) === 'selected_run' ? '' : normalizeQueueScope(filters.queue_scope)],
    ['queueSort', normalizeQueueSort(filters.queue_sort) === 'priority' ? '' : normalizeQueueSort(filters.queue_sort)],
  ];

  for (const [paramKey, value] of mappings) {
    const hasExplicit = params.has(paramKey) && String(params.get(paramKey) || '').trim();
    if (preserveExplicit && hasExplicit) continue;
    if (value) next.set(paramKey, value);
    else next.delete(paramKey);
  }
  return next;
};

const buildSelectedNodeLinks = (node: AgentControlRunNode | null, detail?: AgentControlRunDetail) => {
  if (!node) return [] as Array<{ label: string; path: string }>;
  const links: Array<{ label: string; path: string }> = [];
  const metadata = node.metadata || {};
  const actionPath = String(metadata.action_path || '').trim();
  const notePath = String(metadata.note_path || '').trim();
  const synthesisPath = String(metadata.synthesis_path || '').trim();
  const queuePath = String(metadata.queue_path || '').trim();
  if (node.kind === 'operator_review') {
    if (actionPath) links.push({ label: 'Open operator review', path: actionPath });
    if (queuePath) links.push({ label: 'Open in queue', path: queuePath });
    if (notePath) links.push({ label: 'Open linked note', path: notePath });
    if (synthesisPath) links.push({ label: 'Open linked synthesis', path: synthesisPath });
  }
  if (node.kind === 'agent_job') {
    const jobId = String(metadata.agent_job_id || (node.id.startsWith('job:') ? node.id.slice(4) : '') || '').trim();
    if (jobId) links.push({ label: 'Open in Autonomous Agents', path: `/autonomous-agents?job=${encodeURIComponent(jobId)}` });
  }
  if (node.kind === 'research_note') {
    const noteId = String(metadata.research_note_id || metadata.note_id || '').trim();
    if (noteId) links.push({ label: 'Open in Research Notes', path: `/research-notes?note=${encodeURIComponent(noteId)}` });
  }
  const synthesisJobId = String(metadata.synthesis_job_id || '').trim();
  if (node.kind === 'synthesis_job' || synthesisJobId) {
    if (synthesisJobId) links.push({ label: 'Open in Synthesis', path: `/synthesis?job=${encodeURIComponent(synthesisJobId)}` });
  }
  const workflowExecutionId = String(metadata.workflow_execution_id || '').trim();
  if (node.kind === 'workflow_execution' && workflowExecutionId) {
    links.push({ label: 'Open workflow execution', path: `/workflows?executionId=${encodeURIComponent(workflowExecutionId)}` });
  }
  const routingExperimentId = String(metadata.routing_experiment_id || '').trim();
  const routingVariantId = String(metadata.routing_experiment_variant_id || '').trim();
  const routingProvider = String(metadata.provider || detail?.routing?.provider || '').trim();
  const routingModel = String(metadata.model || detail?.routing?.model || '').trim();
  const routingTier = String(metadata.routing_tier || detail?.routing?.routing_tier || '').trim();
  const routingParams = new URLSearchParams();
  if (routingProvider) routingParams.set('provider', routingProvider);
  if (routingModel) routingParams.set('model', routingModel);
  if (routingTier) routingParams.set('routing_tier', routingTier);
  if (routingExperimentId) routingParams.set('experiment_id', routingExperimentId);
  if (routingVariantId) routingParams.set('variant_id', routingVariantId);
  if (routingParams.toString()) {
    links.push({ label: 'Open routing slice', path: `/usage/routing?${routingParams.toString()}` });
  } else if (detail?.routing?.summary) {
    links.push({ label: 'Open in Routing Observability', path: '/usage/routing' });
  }
  return links;
};

const STAGE_LANES: Array<{ key: string; label: string }> = [
  { key: 'planner', label: 'Planner' },
  { key: 'router', label: 'Router' },
  { key: 'executor', label: 'Executor' },
  { key: 'operator_review', label: 'Operator Review' },
];

const StageCard: React.FC<{ title: string; body?: string | null }> = ({ title, body }) => (
  <div className="rounded-xl border border-gray-200 bg-white p-4">
    <div className="text-xs font-semibold uppercase tracking-[0.16em] text-gray-500">{title}</div>
    <p className="mt-2 text-sm leading-6 text-gray-700">{body || 'No lineage available.'}</p>
  </div>
);

const NodeListItem: React.FC<{ node: AgentControlRunNode; isSelected: boolean; onClick: () => void }> = ({
  node,
  isSelected,
  onClick,
}) => (
  <button
    type="button"
    onClick={onClick}
    className={`w-full rounded-xl border px-4 py-3 text-left transition ${
      isSelected ? 'border-primary-500 bg-primary-500/15 text-primary-700' : 'border-gray-200 bg-white hover:border-gray-300'
    }`}
  >
    <div className="flex items-start justify-between gap-3">
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 ${isSelected ? 'text-gray-700' : 'text-gray-500'}`}>{kindIcon(node.kind)}</div>
        <div>
          <div className="text-sm font-semibold">{node.label}</div>
          <div className={`mt-1 text-xs ${isSelected ? 'text-gray-600' : 'text-gray-500'}`}>
            {node.kind.replace(/_/g, ' ')}{node.stage ? ` • ${node.stage}` : ''}
          </div>
        </div>
      </div>
      {node.status ? (
        <span className={`rounded-full px-2 py-1 text-[11px] font-medium ${isSelected ? 'bg-white/15 text-white' : statusTone(node.status)}`}>
          {node.status}
        </span>
      ) : null}
    </div>
  </button>
);

const AgentControlPlanePage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const sourceType = searchParams.get('type') || '';
  const outcome = searchParams.get('outcome') || '';
  const routingTier = searchParams.get('routingTier') || '';
  const selectedRunId = searchParams.get('run') || '';
  const selectedNodeId = searchParams.get('node') || '';
  const selectedViewId = searchParams.get('view') || '';
  const hasOperatorReview = searchParams.get('hasOperatorReview') || '';
  const reviewType = searchParams.get('reviewType') || '';
  const reviewStatus = searchParams.get('reviewStatus') || '';
  const queueStatus = searchParams.get('queueStatus') || '';
  const queueCustomer = searchParams.get('queueCustomer') || '';
  const queueSla = searchParams.get('queueSla') || '';
  const queueEscalation = searchParams.get('queueEscalation') || '';
  const queueHealthDrilldown = normalizeQueueHealthDrilldown(searchParams.get('queueHealthDrilldown'));
  const queuePreset = normalizeQueuePreset(searchParams.get('queuePreset'));
  const queueScope = normalizeQueueScope(searchParams.get('queueScope'));
  const queueSort = normalizeQueueSort(searchParams.get('queueSort'));
  const queueOffset = parseQueueOffset(searchParams.get('queueOffset'));
  const [viewNameDraft, setViewNameDraft] = useState('');
  const [viewIsDefaultDraft, setViewIsDefaultDraft] = useState(false);
  const [reviewNoteDrafts, setReviewNoteDrafts] = useState<Record<string, string>>({});
  const [checkpointEditDrafts, setCheckpointEditDrafts] = useState<Record<string, { tool: string; purpose: string; params: string }>>({});
  const [reviewSelection, setReviewSelection] = useState<Record<string, boolean>>({});
  const [bulkReviewNote, setBulkReviewNote] = useState('');
  const [globalReviewPages, setGlobalReviewPages] = useState<Record<number, AgentControlRunReviewItem[]>>({});

  const controlRunViewsQuery = useQuery(
    ['agent-control-run-views'],
    () => apiClient.listAgentControlRunViews(),
    {
      staleTime: 30000,
    }
  );

  const runsQuery = useQuery(
    ['agent-control-runs', sourceType, hasOperatorReview, reviewType, reviewStatus],
    () =>
      apiClient.getAgentControlRuns({
        source_type: sourceType || undefined,
        has_operator_review: hasOperatorReview ? hasOperatorReview === 'true' : undefined,
        review_type: reviewType || undefined,
        review_status: reviewStatus || undefined,
        limit: 50,
      }),
    {
      keepPreviousData: true,
    }
  );

  const globalReviewsQuery = useQuery(
    [
      'agent-control-reviews',
      sourceType,
      hasOperatorReview,
      reviewType,
      reviewStatus,
      queueStatus,
      queueCustomer,
      queueSla,
      queueEscalation,
      queueHealthDrilldown,
      queuePreset,
      queueSort,
      queueOffset,
    ],
    () =>
      apiClient.getAgentControlReviews({
        source_type: sourceType || undefined,
        has_operator_review: hasOperatorReview ? hasOperatorReview === 'true' : undefined,
        review_type: reviewType || undefined,
        review_status: reviewStatus || undefined,
        queue_status: queueStatus || undefined,
        queue_customer: queueCustomer || undefined,
        queue_sla: queueSla || undefined,
        queue_escalation: queueEscalation || undefined,
        queue_health_drilldown: queueHealthDrilldown || undefined,
        queue_preset: queuePreset || undefined,
        sort: queueSort,
        offset: queueOffset,
        limit: GLOBAL_QUEUE_PAGE_SIZE,
      }),
    {
      enabled: queueScope === 'global',
      keepPreviousData: true,
    }
  );

  useEffect(() => {
    setGlobalReviewPages({});
  }, [
    queueScope,
    sourceType,
    hasOperatorReview,
    reviewType,
    reviewStatus,
    queueStatus,
    queueCustomer,
    queueSla,
    queueEscalation,
    queueHealthDrilldown,
    queuePreset,
    queueSort,
  ]);

  useEffect(() => {
    if (queueScope !== 'global' || !globalReviewsQuery.data) return;
    const responseOffset = Number(globalReviewsQuery.data.offset ?? 0);
    if (responseOffset !== queueOffset) return;
    setGlobalReviewPages((prev) => {
      const next = queueOffset === 0 ? {} : { ...prev };
      next[queueOffset] = globalReviewsQuery.data.items || [];
      return next;
    });
  }, [queueScope, queueOffset, globalReviewsQuery.data]);

  const filteredRuns = useMemo(() => {
    const items = runsQuery.data?.items || [];
    return items.filter((item) => {
      if (outcome && (item.outcome || '').toLowerCase() !== outcome.toLowerCase()) return false;
      if (routingTier && (item.routing?.routing_tier || '').toLowerCase() !== routingTier.toLowerCase()) return false;
      if (hasOperatorReview === 'true' && Number(item.queued_operator_review_count || 0) <= 0) return false;
      if (hasOperatorReview === 'false' && Number(item.queued_operator_review_count || 0) > 0) return false;
      if (reviewType && Number(item.queued_operator_reviews_by_type?.[reviewType] || 0) <= 0) return false;
      if (reviewStatus && reviewStatus !== 'queued') return false;
      if (queueHealthDrilldown === 'pending_follow_up_approvals' && Number(item.queued_operator_reviews_by_type?.follow_up_recommendation || 0) <= 0) {
        return false;
      }
      if (queueHealthDrilldown === 'manual_follow_up_recommendations' && Number(item.queued_operator_reviews_by_type?.manual_follow_up_recommendation || 0) <= 0) {
        return false;
      }
      if (
        queueHealthDrilldown === 'blocked_follow_up' &&
        Number(item.queued_operator_reviews_by_type?.budget_review || 0) <= 0 &&
        Number(item.queued_operator_reviews_by_type?.policy_review || 0) <= 0
      ) {
        return false;
      }
      if (
        queuePreset === 'approval_required' &&
        Number(item.queued_operator_reviews_by_type?.approval_checkpoint || 0) <= 0 &&
        Number(item.queued_operator_reviews_by_type?.follow_up_recommendation || 0) <= 0
      ) {
        return false;
      }
      if (queuePreset === 'failed_recovery' && Number(item.queued_operator_reviews_by_type?.job_recovery || 0) <= 0) return false;
      if (queuePreset === 'compiler' && !includesCompilerToken(item.title, item.subtitle)) return false;
      return true;
    });
  }, [outcome, routingTier, hasOperatorReview, reviewType, reviewStatus, queueHealthDrilldown, queuePreset, runsQuery.data?.items]);

  useEffect(() => {
    if (filteredRuns.length === 0) return;
    if (selectedRunId && filteredRuns.some((item) => item.id === selectedRunId)) return;
    const next = new URLSearchParams(searchParams);
    next.set('run', filteredRuns[0].id);
    next.delete('node');
    setSearchParams(next, { replace: true });
  }, [filteredRuns, searchParams, selectedRunId, setSearchParams]);

  const detailQuery = useQuery(
    ['agent-control-run', selectedRunId],
    () => apiClient.getAgentControlRun(selectedRunId),
    {
      enabled: Boolean(selectedRunId),
      keepPreviousData: true,
    }
  );

  const selectedNode = useMemo(() => {
    const detail = detailQuery.data;
    if (!detail || !selectedNodeId) return null;
    return detail.nodes.find((node) => node.id === selectedNodeId) || null;
  }, [detailQuery.data, selectedNodeId]);
  const detail: AgentControlRunDetail | undefined = detailQuery.data;
  const filteredReviewItems = useMemo(
    () =>
      (detail?.queued_operator_reviews || []).filter((review) =>
        reviewMatchesQueueFilters({
          review,
          reviewType,
          reviewStatus,
          queueStatus,
          queueCustomer,
          queueSla,
          queueEscalation,
          queueHealthDrilldown,
          queuePreset,
        })
      ),
    [detail?.queued_operator_reviews, reviewType, reviewStatus, queueStatus, queueCustomer, queueSla, queueEscalation, queueHealthDrilldown, queuePreset]
  );

  const globalReviewItems = useMemo(
    () =>
      Object.entries(globalReviewPages)
        .map(([offset, items]) => ({ offset: Number(offset), items }))
        .sort((a, b) => a.offset - b.offset)
        .flatMap((entry) => entry.items),
    [globalReviewPages]
  );
  const globalQueueSummary = globalReviewsQuery.data?.summary || null;
  const globalQueueHasMore = Boolean(globalReviewsQuery.data?.has_more);

  const activeReviewItems = queueScope === 'global' ? globalReviewItems : filteredReviewItems;

  const groupedNodes = useMemo(() => {
    const detail = detailQuery.data;
    if (!detail) return [] as Array<{ key: string; label: string; nodes: AgentControlRunNode[] }>;
    const buckets = new Map<string, AgentControlRunNode[]>();
    for (const lane of STAGE_LANES) buckets.set(lane.key, []);
    const uncategorized: AgentControlRunNode[] = [];
    for (const node of detail.nodes) {
      const key = String(node.stage || '').trim();
      if (buckets.has(key)) buckets.get(key)?.push(node);
      else uncategorized.push(node);
    }
    const groups = STAGE_LANES
      .map((lane) => ({ key: lane.key, label: lane.label, nodes: buckets.get(lane.key) || [] }))
      .filter((lane) => lane.nodes.length > 0);
    if (uncategorized.length > 0) groups.push({ key: 'uncategorized', label: 'Uncategorized', nodes: uncategorized });
    return groups;
  }, [detailQuery.data]);

  const selectedNodeRelations = useMemo(() => {
    if (!detail || !selectedNode) return { inbound: [], outbound: [] } as {
      inbound: Array<{ edge: AgentControlRunEdge; node: AgentControlRunNode | null }>;
      outbound: Array<{ edge: AgentControlRunEdge; node: AgentControlRunNode | null }>;
    };
    const byId = new Map(detail.nodes.map((node) => [node.id, node]));
    const inbound = detail.edges
      .filter((edge) => edge.target === selectedNode.id)
      .map((edge) => ({ edge, node: byId.get(edge.source) || null }));
    const outbound = detail.edges
      .filter((edge) => edge.source === selectedNode.id)
      .map((edge) => ({ edge, node: byId.get(edge.target) || null }));
    return { inbound, outbound };
  }, [detail, selectedNode]);

  const resetGlobalQueueOffset = (params: URLSearchParams) => {
    params.delete('queueOffset');
  };

  const updateFilter = (key: string, value: string) => {
    const next = new URLSearchParams(searchParams);
    if (value) next.set(key, value);
    else next.delete(key);
    if (key !== 'run' && key !== 'queueOffset') next.delete('node');
    if (key !== 'queueOffset') resetGlobalQueueOffset(next);
    setSearchParams(next);
  };

  const selectRun = (run: AgentControlRunSummary) => {
    const next = new URLSearchParams(searchParams);
    next.set('run', run.id);
    next.delete('node');
    setSearchParams(next);
  };

  const selectNode = (nodeId: string) => {
    const next = new URLSearchParams(searchParams);
    next.set('node', nodeId);
    setSearchParams(next);
  };

  const selectedNodeLinks = useMemo(() => buildSelectedNodeLinks(selectedNode, detail), [selectedNode, detail]);
  const currentControlViewFilters = useMemo(
    () => buildCurrentControlViewFilters(searchParams),
    [searchParams]
  );

  const createControlRunViewMutation = useMutation(
    (payload: { name: string; filters: Record<string, any>; is_default?: boolean }) => apiClient.createAgentControlRunView(payload),
    {
      onSuccess: (view) => {
        queryClient.invalidateQueries(['agent-control-run-views']);
        const next = applyControlViewToParams({
          params: searchParams,
          view,
          preserveExplicit: false,
        });
        setSearchParams(next, { replace: true });
        setViewNameDraft(String(view.name || ''));
        setViewIsDefaultDraft(Boolean(view.is_default));
        toast.success('Control-plane view saved');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to save control-plane view');
      },
    }
  );

  const updateControlRunViewMutation = useMutation(
    ({ viewId, payload }: { viewId: string; payload: { name?: string; filters?: Record<string, any>; is_default?: boolean } }) =>
      apiClient.updateAgentControlRunView(viewId, payload),
    {
      onSuccess: (view) => {
        queryClient.invalidateQueries(['agent-control-run-views']);
        setViewNameDraft(String(view.name || ''));
        setViewIsDefaultDraft(Boolean(view.is_default));
        toast.success('Control-plane view updated');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to update control-plane view');
      },
    }
  );

  const deleteControlRunViewMutation = useMutation(
    (viewId: string) => apiClient.deleteAgentControlRunView(viewId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agent-control-run-views']);
        const next = new URLSearchParams(searchParams);
        next.delete('view');
        setSearchParams(next, { replace: true });
        setViewNameDraft('');
        setViewIsDefaultDraft(false);
        toast.success('Control-plane view deleted');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to delete control-plane view');
      },
    }
  );

  const actOnReviewMutation = useMutation(
    (payload: {
      review_type: string;
      source_kind: string;
      source_id: string;
      opportunity_id: string;
      action: string;
      operator_note?: string | null;
      checkpoint_action_patch?: Record<string, any> | null;
    }) => apiClient.actOnAgentControlReview(payload),
    {
      onSuccess: (response, variables) => {
        const reviewKey = buildReviewQueueKey(variables);
        setReviewNoteDrafts((prev) => {
          if (!prev[reviewKey]) return prev;
          const next = { ...prev };
          delete next[reviewKey];
          return next;
        });
        if (variables.action === 'edit') {
          const checkpointDraftKey = buildCheckpointDraftKey(variables);
          setCheckpointEditDrafts((prev) => {
            if (!prev[checkpointDraftKey]) return prev;
            const next = { ...prev };
            delete next[checkpointDraftKey];
            return next;
          });
        }
        queryClient.invalidateQueries(['agent-control-runs']);
        queryClient.invalidateQueries(['agent-control-run']);
        queryClient.invalidateQueries(['agent-control-reviews']);
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['research-monitor-analytics']);
        queryClient.invalidateQueries(['agent-checkpoint-queue']);
        queryClient.invalidateQueries(['research-inbox']);
        toast.success(response.detail || 'Operator review updated');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to update operator review');
      },
    }
  );

  const bulkActOnReviewMutation = useMutation(
    (payload: {
      item_type: string;
      action: string;
      job_ids?: string[];
      domain_research_profile_id?: string | null;
      profile_opportunity_ids?: string[];
      portfolio_id?: string | null;
      portfolio_opportunity_ids?: string[];
      operator_note?: string | null;
    }) => apiClient.bulkActOnAgentControlReview(payload),
    {
      onSuccess: (response) => {
        setReviewSelection({});
        setBulkReviewNote('');
        queryClient.invalidateQueries(['agent-control-runs']);
        queryClient.invalidateQueries(['agent-control-run']);
        queryClient.invalidateQueries(['agent-control-reviews']);
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['agent-jobs']);
        toast.success(
          response.failed > 0
            ? `Bulk action applied to ${response.applied} of ${response.requested_count} items`
            : `Bulk action applied to ${response.applied} items`
        );
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to apply bulk operator review action');
      },
    }
  );

  useEffect(() => {
    const selectedView = (controlRunViewsQuery.data?.items || []).find((item) => item.id === selectedViewId);
    if (!selectedView) return;
    setViewNameDraft(String(selectedView.name || ''));
    setViewIsDefaultDraft(Boolean(selectedView.is_default));
  }, [controlRunViewsQuery.data?.items, selectedViewId]);

  useEffect(() => {
    const items = controlRunViewsQuery.data?.items || [];
    if (items.length === 0) return;

    if (selectedViewId) {
      const selectedView = items.find((item) => item.id === selectedViewId);
      if (!selectedView) return;
      const explicitFilterKeys = CONTROL_VIEW_QUERY_KEYS.filter((key) => {
        if (key === 'run') return searchParams.has(key) && String(searchParams.get(key) || '').trim();
        return searchParams.has(key);
      });
      const next = applyControlViewToParams({
        params: searchParams,
        view: selectedView,
        preserveExplicit: explicitFilterKeys.length > 0,
      });
      if (next.toString() !== searchParams.toString()) {
        if (next.get('run') !== searchParams.get('run')) next.delete('node');
        setSearchParams(next, { replace: true });
      }
      return;
    }

    const hasExplicitFilters = CONTROL_VIEW_QUERY_KEYS.some((key) => {
      if (key === 'run') return searchParams.has(key) && String(searchParams.get(key) || '').trim();
      return searchParams.has(key);
    });
    if (hasExplicitFilters) return;

    const defaultView = items.find((item) => item.is_default);
    if (!defaultView) return;
    const next = applyControlViewToParams({
      params: searchParams,
      view: defaultView,
      preserveExplicit: false,
    });
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [controlRunViewsQuery.data?.items, searchParams, selectedViewId, setSearchParams]);

  useEffect(() => {
    setReviewSelection({});
    setBulkReviewNote('');
  }, [selectedRunId]);

  useEffect(() => {
    const validKeys = new Set(activeReviewItems.map((review) => buildReviewSelectionKey(review)));
    setReviewSelection((prev) => {
      const next = Object.fromEntries(Object.entries(prev).filter(([key, selected]) => selected && validKeys.has(key)));
      if (Object.keys(next).length === Object.keys(prev).length) return prev;
      return next;
    });
  }, [activeReviewItems]);

  const selectedReviewItems = useMemo(
    () => activeReviewItems.filter((review) => reviewSelection[buildReviewSelectionKey(review)]),
    [activeReviewItems, reviewSelection]
  );

  const bulkReviewState = useMemo(() => {
    if (selectedReviewItems.length === 0) {
      return {
        itemType: null as string | null,
        actions: [] as string[],
        disabledReason: 'Select one or more queue items to use bulk triage.',
        profileId: null as string | null,
        portfolioId: null as string | null,
        profileOpportunityIds: [] as string[],
        portfolioOpportunityIds: [] as string[],
        jobIds: [] as string[],
      };
    }
    const bulkClasses = Array.from(new Set(selectedReviewItems.map((item) => normalizeReviewBulkClass(item)).filter(Boolean)));
    if (bulkClasses.length !== 1) {
      return {
        itemType: null as string | null,
        actions: [] as string[],
        disabledReason: 'Selected items must share one bulk-action queue class.',
        profileId: null as string | null,
        portfolioId: null as string | null,
        profileOpportunityIds: [] as string[],
        portfolioOpportunityIds: [] as string[],
        jobIds: [] as string[],
      };
    }

    const itemType = bulkClasses[0];
    if (itemType === 'approval_checkpoint') {
      const jobIds = selectedReviewItems.map((item) => String(item.job_id || '').trim()).filter(Boolean);
      if (jobIds.length !== selectedReviewItems.length) {
        return {
          itemType: null as string | null,
          actions: [] as string[],
          disabledReason: 'Approval checkpoint bulk actions require job-linked items only.',
          profileId: null as string | null,
          portfolioId: null as string | null,
          profileOpportunityIds: [] as string[],
          portfolioOpportunityIds: [] as string[],
          jobIds: [] as string[],
        };
      }
      return {
        itemType,
        actions: ['approve', 'reject', 'skip'],
        disabledReason: '',
        profileId: null as string | null,
        portfolioId: null as string | null,
        profileOpportunityIds: [] as string[],
        portfolioOpportunityIds: [] as string[],
        jobIds,
      };
    }

    if (itemType === 'job_recovery') {
      const jobIds = selectedReviewItems.map((item) => String(item.job_id || '').trim()).filter(Boolean);
      if (jobIds.length !== selectedReviewItems.length) {
        return {
          itemType: null as string | null,
          actions: [] as string[],
          disabledReason: 'Recovery bulk actions require job-linked items only.',
          profileId: null as string | null,
          portfolioId: null as string | null,
          profileOpportunityIds: [] as string[],
          portfolioOpportunityIds: [] as string[],
          jobIds: [] as string[],
        };
      }
      return {
        itemType,
        actions: ['restart', 'resume', 'cancel'],
        disabledReason: '',
        profileId: null as string | null,
        portfolioId: null as string | null,
        profileOpportunityIds: [] as string[],
        portfolioOpportunityIds: [] as string[],
        jobIds,
      };
    }

    if (itemType === 'follow_up_recommendation') {
      const ownerKinds = Array.from(new Set(selectedReviewItems.map((item) => String(item.source_kind || '').trim()).filter(Boolean)));
      const ownerIds = Array.from(new Set(selectedReviewItems.map((item) => String(item.source_id || '').trim()).filter(Boolean)));
      if (ownerKinds.length !== 1 || ownerIds.length !== 1 || !['profile', 'portfolio'].includes(ownerKinds[0])) {
        return {
          itemType: null as string | null,
          actions: [] as string[],
          disabledReason: 'Bulk follow-up approvals must stay within one domain profile or research fleet.',
          profileId: null as string | null,
          portfolioId: null as string | null,
          profileOpportunityIds: [] as string[],
          portfolioOpportunityIds: [] as string[],
          jobIds: [] as string[],
        };
      }
      const opportunityIds = selectedReviewItems.map((item) => String(item.opportunity_id || '').trim()).filter(Boolean);
      if (opportunityIds.length !== selectedReviewItems.length) {
        return {
          itemType: null as string | null,
          actions: [] as string[],
          disabledReason: 'Bulk follow-up approvals require opportunity-linked queue items.',
          profileId: null as string | null,
          portfolioId: null as string | null,
          profileOpportunityIds: [] as string[],
          portfolioOpportunityIds: [] as string[],
          jobIds: [] as string[],
        };
      }
      return {
        itemType,
        actions: ['approve_launch', 'reject_launch'],
        disabledReason: '',
        profileId: ownerKinds[0] === 'profile' ? ownerIds[0] : null,
        portfolioId: ownerKinds[0] === 'portfolio' ? ownerIds[0] : null,
        profileOpportunityIds: ownerKinds[0] === 'profile' ? opportunityIds : [],
        portfolioOpportunityIds: ownerKinds[0] === 'portfolio' ? opportunityIds : [],
        jobIds: [] as string[],
      };
    }

    return {
      itemType: null as string | null,
      actions: [] as string[],
      disabledReason: 'Selected items do not support bulk actions.',
      profileId: null as string | null,
      portfolioId: null as string | null,
      profileOpportunityIds: [] as string[],
      portfolioOpportunityIds: [] as string[],
      jobIds: [] as string[],
    };
  }, [selectedReviewItems]);

  const selectVisibleReviewItems = () => {
    if (!activeReviewItems.length) return;
    setReviewSelection((prev) => {
      const next = { ...prev };
      for (const review of activeReviewItems) {
        next[buildReviewSelectionKey(review)] = true;
      }
      return next;
    });
  };

  const clearReviewSelection = () => setReviewSelection({});

  const getCheckpointDraftValue = (review: AgentControlRunReviewItem) => {
    const key = buildCheckpointDraftKey(review);
    return checkpointEditDrafts[key] || getInitialCheckpointDraft(review);
  };

  const updateCheckpointDraftValue = (
    review: AgentControlRunReviewItem,
    field: 'tool' | 'purpose' | 'params',
    value: string
  ) => {
    const key = buildCheckpointDraftKey(review);
    const current = getCheckpointDraftValue(review);
    setCheckpointEditDrafts((prev) => ({
      ...prev,
      [key]: {
        ...current,
        [field]: value,
      },
    }));
  };

  const toggleReviewSelection = (review: { queue_item_key?: string | null; source_kind?: string | null; source_id?: string | null; opportunity_id?: string | null; review_type?: string | null; }) => {
    const key = buildReviewSelectionKey(review);
    setReviewSelection((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-500">Agent Control Plane</div>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-gray-900">Planner, router, and executor lineage</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-gray-600">
              Read-only control runs stitched from autonomous jobs, workflow executions, decision trace events, and task memory.
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <label className="text-sm">
              <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">Source</div>
              <select className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2" value={sourceType} onChange={(event) => updateFilter('type', event.target.value)}>
                <option value="">All</option>
                <option value="job">Jobs</option>
                <option value="workflow">Workflows</option>
              </select>
            </label>
            <label className="text-sm">
              <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">Outcome</div>
              <select className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2" value={outcome} onChange={(event) => updateFilter('outcome', event.target.value)}>
                <option value="">All</option>
                <option value="running">Running</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="pending">Pending</option>
              </select>
            </label>
            <label className="text-sm">
              <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">Routing tier</div>
              <select className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2" value={routingTier} onChange={(event) => updateFilter('routingTier', event.target.value)}>
                <option value="">All</option>
                <option value="fast">fast</option>
                <option value="balanced">balanced</option>
                <option value="premium">premium</option>
              </select>
            </label>
            <label className="text-sm">
              <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">Operator review</div>
              <select className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2" value={hasOperatorReview} onChange={(event) => updateFilter('hasOperatorReview', event.target.value)}>
                <option value="">All</option>
                <option value="true">Queued only</option>
                <option value="false">No queued reviews</option>
              </select>
            </label>
            <label className="text-sm">
              <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">Review type</div>
              <select className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2" value={reviewType} onChange={(event) => updateFilter('reviewType', event.target.value)}>
                <option value="">All</option>
                <option value="follow_up_recommendation">follow_up_recommendation</option>
                <option value="manual_follow_up_recommendation">manual_follow_up_recommendation</option>
                <option value="approval_checkpoint">approval_checkpoint</option>
                <option value="job_recovery">job_recovery</option>
                <option value="budget_review">budget_review</option>
                <option value="policy_review">policy_review</option>
              </select>
            </label>
            <label className="text-sm">
              <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">Review status</div>
              <select className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2" value={reviewStatus} onChange={(event) => updateFilter('reviewStatus', event.target.value)}>
                <option value="">All</option>
                <option value="queued">queued</option>
              </select>
            </label>
            <label className="text-sm">
              <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">Queue status</div>
              <input
                className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2"
                value={queueStatus}
                onChange={(event) => updateFilter('queueStatus', event.target.value)}
                placeholder="pending_approval, paused, blocked"
              />
            </label>
            <label className="text-sm">
              <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">Queue customer</div>
              <input
                className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2"
                value={queueCustomer}
                onChange={(event) => updateFilter('queueCustomer', event.target.value)}
                placeholder="compiler"
              />
            </label>
            <label className="text-sm">
              <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">Queue SLA</div>
              <input
                className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2"
                value={queueSla}
                onChange={(event) => updateFilter('queueSla', event.target.value)}
                placeholder="at_risk, overdue"
              />
            </label>
            <label className="text-sm">
              <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">Escalation</div>
              <input
                className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2"
                value={queueEscalation}
                onChange={(event) => updateFilter('queueEscalation', event.target.value)}
                placeholder="medium, high"
              />
            </label>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            {([
              ['selected_run', 'Selected run queue'],
              ['global', 'Global queue'],
            ] as Array<[QueueScope, string]>).map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => updateFilter('queueScope', value === 'selected_run' ? '' : value)}
                className={`rounded-full border px-3 py-1.5 text-sm font-medium ${
                  queueScope === value
                    ? 'border-emerald-900 bg-emerald-900 text-white'
                    : 'border-emerald-200 bg-emerald-50 text-emerald-700 hover:border-emerald-500'
                }`}
              >
                {label}
              </button>
            ))}
            {([
              ['pending_follow_up_approvals', 'Pending approvals'],
              ['manual_follow_up_recommendations', 'Manual recommendations'],
              ['blocked_follow_up', 'Blocked follow-ups'],
            ] as Array<[QueueHealthDrilldown, string]>).map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => updateFilter('queueHealthDrilldown', queueHealthDrilldown === value ? '' : value)}
                className={`rounded-full border px-3 py-1.5 text-sm font-medium ${
                  queueHealthDrilldown === value
                    ? 'border-primary-500 bg-primary-500/15 text-primary-700'
                    : 'border-gray-300 bg-white text-gray-700 hover:border-primary-500'
                }`}
              >
                {label}
              </button>
            ))}
            {([
              ['approval_required', 'Approval required'],
              ['failed_recovery', 'Failed recovery'],
              ['compiler', 'Compiler'],
            ] as Array<[QueuePreset, string]>).map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => updateFilter('queuePreset', queuePreset === value ? '' : value)}
                className={`rounded-full border px-3 py-1.5 text-sm font-medium ${
                  queuePreset === value
                    ? 'border-sky-900 bg-sky-900 text-white'
                    : 'border-sky-200 bg-sky-50 text-sky-700 hover:border-sky-500'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-center gap-3">
          <select
            aria-label="Control-plane saved view"
            className="rounded-lg border border-gray-300 px-3 py-2 text-sm"
            value={selectedViewId}
            onChange={(event) => {
              const viewId = event.target.value;
              if (!viewId) {
                const next = new URLSearchParams(searchParams);
                next.delete('view');
                setSearchParams(next);
                setViewNameDraft('');
                setViewIsDefaultDraft(false);
                return;
              }
              const nextView = (controlRunViewsQuery.data?.items || []).find((item) => item.id === viewId);
              if (!nextView) return;
              const next = applyControlViewToParams({
                params: new URLSearchParams(),
                view: nextView,
                preserveExplicit: false,
              });
              next.delete('node');
              setSearchParams(next);
            }}
          >
            <option value="">Saved views</option>
            {(controlRunViewsQuery.data?.items || []).map((view) => (
              <option key={view.id} value={view.id}>
                {view.name}{view.is_default ? ' (Default)' : ''}
              </option>
            ))}
          </select>
          <input
            aria-label="Control-plane view name"
            className="min-w-[220px] rounded-lg border border-gray-300 px-3 py-2 text-sm"
            placeholder="Current view name"
            value={viewNameDraft}
            onChange={(event) => setViewNameDraft(event.target.value)}
          />
          <label className="inline-flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-100 px-3 py-2 text-sm text-gray-700">
            <input
              aria-label="Default control-plane view"
              type="checkbox"
              className="h-4 w-4 rounded border-gray-300"
              checked={viewIsDefaultDraft}
              onChange={(event) => setViewIsDefaultDraft(event.target.checked)}
            />
            <span>Default view</span>
          </label>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              const name = String(viewNameDraft || '').trim();
              if (!name) {
                toast.error('Name the control-plane view first');
                return;
              }
              createControlRunViewMutation.mutate({
                name,
                filters: currentControlViewFilters,
                is_default: viewIsDefaultDraft,
              });
            }}
          >
            Save Current View
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={!selectedViewId}
            onClick={() => {
              if (!selectedViewId) return;
              updateControlRunViewMutation.mutate({
                viewId: selectedViewId,
                payload: {
                  name: String(viewNameDraft || '').trim() || undefined,
                  filters: currentControlViewFilters,
                  is_default: viewIsDefaultDraft,
                },
              });
            }}
          >
            Update View
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={!selectedViewId}
            onClick={() => {
              if (!selectedViewId) return;
              deleteControlRunViewMutation.mutate(selectedViewId);
            }}
          >
            Delete View
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              const next = new URLSearchParams();
              setSearchParams(next);
              setViewNameDraft('');
              setViewIsDefaultDraft(false);
            }}
          >
            Reset
          </Button>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)_380px]">
        <section className="rounded-2xl border border-gray-200 bg-gray-100 p-4 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-gray-900">Runs</h2>
              <p className="text-xs text-gray-500">{filteredRuns.length} visible</p>
            </div>
            {runsQuery.isFetching ? <LoadingSpinner size="sm" /> : null}
          </div>
          <div className="space-y-3">
            {(filteredRuns || []).map((run) => (
              <button
                type="button"
                key={run.id}
                onClick={() => selectRun(run)}
                className={`w-full rounded-xl border px-4 py-3 text-left transition ${
                  selectedRunId === run.id ? 'border-primary-500 bg-primary-500/15 text-primary-700' : 'border-gray-200 bg-white hover:border-gray-300'
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold">{run.title}</div>
                    <div className={`mt-1 text-xs ${selectedRunId === run.id ? 'text-gray-600' : 'text-gray-500'}`}>
                      {run.source_type} • {run.replayability_status.replace(/_/g, ' ')}
                    </div>
                  </div>
                  <span className={`rounded-full px-2 py-1 text-[11px] font-medium ${selectedRunId === run.id ? 'bg-white/15 text-white' : statusTone(run.status)}`}>
                    {run.status}
                  </span>
                </div>
                <div className={`mt-3 grid grid-cols-3 gap-2 text-[11px] ${selectedRunId === run.id ? 'text-gray-600' : 'text-gray-500'}`}>
                  <div>{run.child_job_count + run.child_execution_count} downstream</div>
                  <div>{run.linked_note_count} notes</div>
                  <div>{run.linked_experiment_count} experiments</div>
                </div>
                {Number(run.queued_operator_review_count || 0) > 0 ? (
                  <div className={`mt-3 flex flex-wrap gap-2 text-[11px] ${selectedRunId === run.id ? 'text-gray-700' : 'text-gray-600'}`}>
                    <span className={`rounded-full px-2 py-1 font-medium ${selectedRunId === run.id ? 'bg-amber-400/20 text-amber-100' : 'bg-amber-100 text-amber-800'}`}>
                      {run.queued_operator_review_count} queued reviews
                    </span>
                    {Object.entries(run.queued_operator_reviews_by_type || {}).map(([type, count]) => (
                      <span key={`${run.id}-${type}`} className={`rounded-full px-2 py-1 ${selectedRunId === run.id ? 'bg-white/10 text-gray-800' : 'bg-gray-200 text-gray-700'}`}>
                        {type}: {count}
                      </span>
                    ))}
                  </div>
                ) : null}
              </button>
            ))}
            {!runsQuery.isLoading && filteredRuns.length === 0 ? (
              <div className="rounded-xl border border-dashed border-gray-300 bg-white px-4 py-8 text-center text-sm text-gray-500">
                No control runs match the current filters.
              </div>
            ) : null}
          </div>
        </section>

        <section className="space-y-6">
          <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
            {!selectedRunId && !runsQuery.isLoading ? (
              <div className="flex min-h-[240px] items-center justify-center text-sm text-gray-500">
                Select a control run to inspect its graph and replay.
              </div>
            ) : detailQuery.isLoading ? (
              <div className="flex min-h-[240px] items-center justify-center">
                <LoadingSpinner size="lg" />
              </div>
            ) : detailQuery.isError ? (
              <div className="flex min-h-[240px] items-center justify-center text-sm text-rose-600">
                This control run could not be loaded.
              </div>
            ) : detail ? (
              <>
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-500">{detail.run.source_type}</div>
                    <h2 className="mt-2 text-2xl font-semibold tracking-tight text-gray-900">{detail.run.title}</h2>
                    <p className="mt-2 text-sm leading-6 text-gray-600">{detail.run.subtitle || 'No root objective summary was recorded.'}</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className={`rounded-full px-3 py-1 text-xs font-medium ${statusTone(detail.run.status)}`}>{detail.run.status}</span>
                    <span className="rounded-full bg-gray-200 px-3 py-1 text-xs font-medium text-gray-700">
                      {detail.run.replayability_status.replace(/_/g, ' ')}
                    </span>
                  </div>
                </div>

                <div className="mt-6 grid gap-4 md:grid-cols-3">
                  <StageCard title="Planner" body={detail.replay.planner_summary} />
                  <StageCard title="Router" body={detail.replay.router_summary} />
                  <StageCard title="Executor" body={detail.replay.executor_summary} />
                </div>

                <div className="mt-6 grid gap-4 md:grid-cols-4">
                  <div className="rounded-xl border border-gray-200 bg-gray-100 p-4">
                    <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Created</div>
                    <div className="mt-2 text-sm text-gray-800">{formatDateTime(detail.run.created_at)}</div>
                  </div>
                  <div className="rounded-xl border border-gray-200 bg-gray-100 p-4">
                    <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Routing</div>
                    <div className="mt-2 text-sm text-gray-800">{detail.routing?.summary || 'No routing snapshot'}</div>
                  </div>
                  <div className="rounded-xl border border-gray-200 bg-gray-100 p-4">
                    <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Memory graph</div>
                    <div className="mt-2 text-sm text-gray-800">{detail.memory_graph?.stats?.memory_count ?? 0} nodes</div>
                  </div>
                  <div className="rounded-xl border border-gray-200 bg-gray-100 p-4">
                    <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Decision trace</div>
                    <div className="mt-2 text-sm text-gray-800">{detail.decision_trace.length} persisted events</div>
                  </div>
                </div>
              </>
            ) : null}
          </div>

          {detail ? (
            <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
              <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-gray-900">Graph + Timeline</h3>
                    <p className="text-xs text-gray-500">Derived nodes and edges for the current run</p>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <GitFork className="h-4 w-4" />
                    {detail.edges.length} edges
                  </div>
                </div>
                <div className="space-y-5">
                  {groupedNodes.map((group) => (
                    <div key={group.key} className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-gray-500">{group.label}</div>
                        <div className="text-xs text-gray-500">{group.nodes.length} nodes</div>
                      </div>
                      <div className="space-y-3">
                        {group.nodes.map((node) => (
                          <NodeListItem
                            key={node.id}
                            node={node}
                            isSelected={selectedNodeId === node.id}
                            onClick={() => selectNode(node.id)}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex items-center gap-2">
                  <Route className="h-4 w-4 text-gray-500" />
                  <h3 className="text-sm font-semibold text-gray-900">Node Inspector</h3>
                </div>
                {selectedNode ? (
                  <div className="space-y-4">
                    <div>
                      <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">{selectedNode.kind.replace(/_/g, ' ')}</div>
                      <div className="mt-1 text-lg font-semibold text-gray-900">{selectedNode.label}</div>
                    </div>
                    <dl className="space-y-3 text-sm">
                      <div>
                        <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">Stage</dt>
                        <dd className="mt-1 text-gray-800">{selectedNode.stage || '—'}</dd>
                      </div>
                      <div>
                        <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">Timestamp</dt>
                        <dd className="mt-1 text-gray-800">{formatDateTime(selectedNode.timestamp)}</dd>
                      </div>
                      <div>
                        <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">Metadata</dt>
                        <dd className="mt-2 overflow-x-auto rounded-xl bg-gray-100 border border-gray-200 p-3 text-xs leading-6 text-gray-800">
                          <pre>{JSON.stringify(selectedNode.metadata || {}, null, 2)}</pre>
                        </dd>
                      </div>
                      <div>
                        <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">Open in</dt>
                        <dd className="mt-2 flex flex-wrap gap-2">
                          {selectedNodeLinks.map((link) => (
                            <Link
                              key={`${selectedNode.id}-${link.path}`}
                              to={link.path}
                              className="rounded-full border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-primary-500 hover:text-gray-900"
                            >
                              {link.label}
                            </Link>
                          ))}
                          {selectedNodeLinks.length === 0 ? (
                            <span className="text-sm text-gray-500">No supported destination for this node yet.</span>
                          ) : null}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">Relations</dt>
                        <dd className="mt-2 space-y-3">
                          <div>
                            <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Inbound</div>
                            <div className="mt-2 space-y-2">
                              {selectedNodeRelations.inbound.map(({ edge, node }) => (
                                <div key={`${edge.source}-${edge.target}-${edge.relation}`} className="rounded-lg border border-gray-200 bg-gray-100 px-3 py-2">
                                  <div className="text-sm font-medium text-gray-800">{node?.label || edge.source}</div>
                                  <div className="text-xs text-gray-500">{edge.relation.replace(/_/g, ' ')}</div>
                                </div>
                              ))}
                              {selectedNodeRelations.inbound.length === 0 ? (
                                <div className="text-sm text-gray-500">No inbound relations.</div>
                              ) : null}
                            </div>
                          </div>
                          <div>
                            <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Outbound</div>
                            <div className="mt-2 space-y-2">
                              {selectedNodeRelations.outbound.map(({ edge, node }) => (
                                <div key={`${edge.source}-${edge.target}-${edge.relation}`} className="rounded-lg border border-gray-200 bg-gray-100 px-3 py-2">
                                  <div className="text-sm font-medium text-gray-800">{node?.label || edge.target}</div>
                                  <div className="text-xs text-gray-500">{edge.relation.replace(/_/g, ' ')}</div>
                                </div>
                              ))}
                              {selectedNodeRelations.outbound.length === 0 ? (
                                <div className="text-sm text-gray-500">No outbound relations.</div>
                              ) : null}
                            </div>
                          </div>
                        </dd>
                      </div>
                    </dl>
                  </div>
                ) : (
                  <div className="text-sm text-gray-500">Select a node to inspect its metadata and stage context.</div>
                )}
              </div>
            </div>
          ) : null}
        </section>

        <section className="space-y-6">
          <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center gap-2">
              <MemoryStick className="h-4 w-4 text-gray-500" />
              <h3 className="text-sm font-semibold text-gray-900">Inspector</h3>
            </div>
            {detail ? (
              <div className="space-y-5">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Related surfaces</div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {detail.related_links.map((link) => (
                      <Link
                        key={`${link.label}-${link.path}`}
                        to={link.path}
                        className="rounded-full border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-primary-500 hover:text-gray-900"
                      >
                        {link.label}
                      </Link>
                    ))}
                    {detail.related_links.length === 0 ? (
                      <div className="text-sm text-gray-500">No linked operator surfaces were recorded for this run.</div>
                    ) : null}
                  </div>
                </div>

                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Policy snapshot</div>
                  <div className="mt-2 rounded-xl bg-gray-100 p-4 text-sm text-gray-700">
                    <div>Automation profile: {detail.run.automation_profile || '—'}</div>
                    <div className="mt-2">Effective policy: {detail.policy_summary?.effective_policy ? 'Attached' : 'Not recorded'}</div>
                  </div>
                </div>

                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Operator review queue</div>
                  <div className="mt-3 space-y-3">
                    <div className="rounded-xl border border-gray-200 bg-gray-100 p-3">
                      <div className="flex flex-col gap-3">
                        <div className="flex flex-wrap items-center gap-2 text-xs text-gray-600">
                          <span className="rounded-full bg-white px-2 py-1 font-medium text-gray-700">
                            {queueScope === 'global' ? 'Global queue' : 'Selected run queue'}
                          </span>
                          <span>
                            Selected {selectedReviewItems.length} of {activeReviewItems.length}
                          </span>
                          {queueScope === 'global' && globalReviewsQuery.isFetching ? <LoadingSpinner size="sm" /> : null}
                        </div>
                        {queueScope === 'global' ? (
                          <div className="rounded-lg border border-gray-200 bg-white p-3">
                            <div className="flex flex-col gap-3">
                              <div className="flex flex-wrap items-center gap-2 text-xs text-gray-600">
                                <span className="rounded-full bg-gray-200 px-2 py-1 font-medium text-gray-700">
                                  Total {globalQueueSummary?.total ?? globalReviewsQuery.data?.total ?? globalReviewItems.length}
                                </span>
                                {Object.entries(globalQueueSummary?.by_type || {}).map(([value, count]) => (
                                  <span key={`global-type-${value}`} className="rounded-full bg-blue-50 px-2 py-1 text-blue-800">
                                    {value.replace(/_/g, ' ')} {count}
                                  </span>
                                ))}
                                {Object.entries(globalQueueSummary?.by_sla_bucket || {}).map(([value, count]) => (
                                  <span key={`global-sla-${value}`} className="rounded-full bg-amber-50 px-2 py-1 text-amber-800">
                                    {value.replace(/_/g, ' ')} {count}
                                  </span>
                                ))}
                              </div>
                              <div className="flex flex-wrap items-center gap-2">
                                <label className="text-xs font-medium text-gray-600">
                                  Sort
                                  <select
                                    aria-label="Global queue sort"
                                    className="ml-2 rounded-lg border border-gray-300 bg-white px-2 py-1 text-sm text-gray-700"
                                    value={queueSort}
                                    onChange={(event) => updateFilter('queueSort', event.target.value === 'priority' ? '' : event.target.value)}
                                  >
                                    <option value="priority">Priority</option>
                                    <option value="created_at_desc">Newest</option>
                                    <option value="age_desc">Oldest first</option>
                                  </select>
                                </label>
                                {globalQueueHasMore ? (
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    disabled={globalReviewsQuery.isFetching}
                                    onClick={() => updateFilter('queueOffset', String(queueOffset + GLOBAL_QUEUE_PAGE_SIZE))}
                                  >
                                    Load more
                                  </Button>
                                ) : null}
                              </div>
                            </div>
                          </div>
                        ) : null}
                        <div className="flex flex-wrap items-center gap-2">
                          <Button size="sm" variant="ghost" onClick={selectVisibleReviewItems}>
                            Select visible
                          </Button>
                          <Button size="sm" variant="ghost" onClick={clearReviewSelection}>
                            Clear selection
                          </Button>
                        </div>
                        {queueHealthDrilldown || queuePreset ? (
                          <div className="rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-800">
                            Showing{' '}
                            {queueHealthDrilldown ? queueDrilldownLabel(queueHealthDrilldown) : queuePresetLabel(queuePreset)}
                            {' '}slice
                            <button
                              type="button"
                              className="ml-2 font-medium underline underline-offset-2"
                              onClick={() => {
                                const next = new URLSearchParams(searchParams);
                                next.delete('queueHealthDrilldown');
                                next.delete('queuePreset');
                                next.delete('queueOffset');
                                next.delete('node');
                                setSearchParams(next);
                              }}
                            >
                              Clear
                            </button>
                          </div>
                        ) : null}
                        {bulkReviewState.itemType ? (
                          <div className="flex flex-wrap items-center gap-2">
                            <input
                              className="min-w-[220px] rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                              placeholder={
                                bulkReviewState.itemType === 'follow_up_recommendation'
                                  ? 'Shared note for selected follow-up approvals'
                                  : bulkReviewState.itemType === 'approval_checkpoint'
                                    ? 'Shared note for selected approvals'
                                    : 'Shared operator note'
                              }
                              value={bulkReviewNote}
                              onChange={(event) => setBulkReviewNote(event.target.value)}
                            />
                            {bulkReviewState.actions.map((action) => (
                              <Button
                                key={`bulk-review-${action}`}
                                size="sm"
                                variant={action === 'reject' || action === 'cancel' || action === 'reject_launch' ? 'ghost' : 'primary'}
                                disabled={bulkActOnReviewMutation.isLoading}
                                onClick={() =>
                                  bulkActOnReviewMutation.mutate({
                                    item_type: bulkReviewState.itemType || '',
                                    action,
                                    job_ids: bulkReviewState.jobIds,
                                    domain_research_profile_id: bulkReviewState.profileId,
                                    profile_opportunity_ids: bulkReviewState.profileOpportunityIds,
                                    portfolio_id: bulkReviewState.portfolioId,
                                    portfolio_opportunity_ids: bulkReviewState.portfolioOpportunityIds,
                                    operator_note: bulkReviewNote.trim() || undefined,
                                  })
                                }
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
                          <div className="text-xs text-gray-600">{bulkReviewState.disabledReason}</div>
                        )}
                      </div>
                    </div>
                    {activeReviewItems.map((review, index) => (
                      <div key={`${review.source_kind || 'review'}-${review.source_id || 'unknown'}-${review.opportunity_id || index}`} className="rounded-xl border border-gray-200 bg-gray-100 p-3">
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex min-w-0 items-start gap-3">
                            <div className="pt-1">
                              <input
                                type="checkbox"
                                className="rounded border-gray-300"
                                checked={!!reviewSelection[buildReviewSelectionKey(review)]}
                                onChange={() => toggleReviewSelection(review)}
                                aria-label={`Select control-plane review ${review.title || review.reason_label || review.review_type || index}`}
                              />
                            </div>
                            <div>
                            <div className="text-sm font-medium text-gray-900">{review.title || review.reason_label || review.review_type || 'Operator review'}</div>
                            <div className="mt-1 text-xs text-gray-500">
                              {review.review_type || 'review'} • {review.review_status || 'queued'} • {review.reason_code || 'no reason code'}
                              {review.status ? ` • status ${review.status}` : ''}
                              {review.follow_up_launch_status ? ` • launch ${review.follow_up_launch_status}` : ''}
                              {review.follow_up_review_status ? ` • review ${review.follow_up_review_status}` : ''}
                            </div>
                          </div>
                          </div>
                          <span className="rounded-full bg-amber-100 px-2 py-1 text-[11px] font-medium text-amber-800">
                            {review.item_type || review.source_kind || 'review'}
                          </span>
                        </div>
                        {review.summary ? (
                          <div className="mt-2 text-sm text-gray-600">{review.summary}</div>
                        ) : null}
                        {review.evidence_summary ? (
                          <div className="mt-2 text-xs text-gray-500">Evidence: {review.evidence_summary}</div>
                        ) : null}
                        {(review.customer || review.job_name || review.job_type) ? (
                          <div className="mt-2 text-xs text-gray-500">
                            {review.customer ? <span>Customer: {review.customer}</span> : null}
                            {review.customer && review.job_name ? <span> • </span> : null}
                            {review.job_name ? <span>Job: {review.job_name}</span> : null}
                            {(review.customer || review.job_name) && review.job_type ? <span> • </span> : null}
                            {review.job_type ? <span>Type: {review.job_type}</span> : null}
                          </div>
                        ) : null}
                        {queueScope === 'global' && review.run_id ? (
                          <div className="mt-2 text-xs text-gray-500">
                            Run: {review.run_title || review.run_id}
                            {review.run_source_type ? <span> • {review.run_source_type}</span> : null}
                            {review.run_status ? <span> • {review.run_status}</span> : null}
                          </div>
                        ) : null}
                        {(review.age_minutes !== undefined && review.age_minutes !== null) || review.priority_score !== undefined || review.sla_bucket || review.escalation_level ? (
                          <div className="mt-2 text-xs text-gray-500">
                            {review.age_minutes !== undefined && review.age_minutes !== null ? <span>Age: {review.age_minutes}m</span> : null}
                            {review.age_minutes !== undefined && review.age_minutes !== null && review.priority_score !== undefined && review.priority_score !== null ? <span> • </span> : null}
                            {review.priority_score !== undefined && review.priority_score !== null ? <span>Urgency: {review.priority_score}</span> : null}
                            {((review.age_minutes !== undefined && review.age_minutes !== null) || (review.priority_score !== undefined && review.priority_score !== null)) && review.sla_bucket ? <span> • </span> : null}
                            {review.sla_bucket ? <span>SLA: {review.sla_bucket.replace(/_/g, ' ')}</span> : null}
                            {(((review.age_minutes !== undefined && review.age_minutes !== null) || (review.priority_score !== undefined && review.priority_score !== null) || review.sla_bucket) && review.escalation_level) ? <span> • </span> : null}
                            {review.escalation_level ? <span>Escalation: {review.escalation_level}</span> : null}
                          </div>
                        ) : null}
                        {review.item_type === 'approval_checkpoint' && review.checkpoint?.action?.tool ? (
                          <div className="mt-2 text-xs text-gray-500">
                            Pending tool: <span className="font-mono">{String(review.checkpoint.action.tool)}</span>
                          </div>
                        ) : null}
                        {review.item_type === 'approval_checkpoint' && ((review.available_actions || []).includes('edit') || review.can_approve) ? (
                          <div className="mt-3 rounded-lg border border-gray-200 bg-white p-3">
                            <div className="text-xs font-medium uppercase tracking-wide text-gray-500">Checkpoint edit</div>
                            <div className="mt-3 grid gap-3 md:grid-cols-2">
                              <label className="text-xs font-medium text-gray-600">
                                Tool
                                <input
                                  aria-label={`Checkpoint tool ${review.title || review.queue_item_key || review.source_id || 'review'}`}
                                  className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-700"
                                  value={getCheckpointDraftValue(review).tool}
                                  onChange={(event) => updateCheckpointDraftValue(review, 'tool', event.target.value)}
                                  placeholder="web.search"
                                />
                              </label>
                              <label className="text-xs font-medium text-gray-600">
                                Purpose
                                <input
                                  aria-label={`Checkpoint purpose ${review.title || review.queue_item_key || review.source_id || 'review'}`}
                                  className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-700"
                                  value={getCheckpointDraftValue(review).purpose}
                                  onChange={(event) => updateCheckpointDraftValue(review, 'purpose', event.target.value)}
                                  placeholder="Refine compiler regression hypothesis"
                                />
                              </label>
                            </div>
                            <label className="mt-3 block text-xs font-medium text-gray-600">
                              Params JSON
                              <textarea
                                aria-label={`Checkpoint params ${review.title || review.queue_item_key || review.source_id || 'review'}`}
                                className="mt-1 min-h-[120px] w-full rounded-lg border border-gray-300 bg-white px-3 py-2 font-mono text-xs text-gray-700"
                                value={getCheckpointDraftValue(review).params}
                                onChange={(event) => updateCheckpointDraftValue(review, 'params', event.target.value)}
                                placeholder='{"q":"compiler regression"}'
                              />
                            </label>
                          </div>
                        ) : null}
                        {review.item_type === 'job_recovery' && summarizeSchedulerState(review.scheduler_state).length > 0 ? (
                          <div className="mt-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-xs text-gray-600">
                            {summarizeSchedulerState(review.scheduler_state).slice(0, 4).map((line) => (
                              <div key={`${review.queue_item_key || review.source_id || 'review'}-${line}`}>{line}</div>
                            ))}
                          </div>
                        ) : null}
                        {review.follow_up_recommendation_key || review.recommendation_score !== undefined ? (
                          <div className="mt-2 text-xs text-gray-500">
                            {review.follow_up_recommendation_key ? <span>Recommendation key: {review.follow_up_recommendation_key}</span> : null}
                            {review.follow_up_recommendation_key && review.recommendation_score !== undefined ? <span> • </span> : null}
                            {review.recommendation_score !== undefined ? <span>Score: {review.recommendation_score}</span> : null}
                          </div>
                        ) : null}
                        {review.follow_up_block_reason ? (
                          <div className="mt-2 text-xs text-amber-700">Blocked: {review.follow_up_block_reason}</div>
                        ) : null}
                        {(review.follow_up_budget_decision || review.follow_up_budget_reason || review.follow_up_customer_budget_decision || review.follow_up_customer_budget_reason) ? (
                          <div className="mt-2 text-xs text-gray-500">
                            {review.follow_up_budget_decision ? <span>Budget: {review.follow_up_budget_decision}</span> : null}
                            {review.follow_up_budget_decision && review.follow_up_budget_reason ? <span> • </span> : null}
                            {review.follow_up_budget_reason ? <span>{review.follow_up_budget_reason}</span> : null}
                            {(review.follow_up_budget_decision || review.follow_up_budget_reason) && review.follow_up_customer_budget_decision ? <span> • </span> : null}
                            {review.follow_up_customer_budget_decision ? <span>Customer budget: {review.follow_up_customer_budget_decision}</span> : null}
                            {review.follow_up_customer_budget_decision && review.follow_up_customer_budget_reason ? <span> • </span> : null}
                            {review.follow_up_customer_budget_reason ? <span>{review.follow_up_customer_budget_reason}</span> : null}
                          </div>
                        ) : null}
                        {review.policy_guardrail_action || (review.policy_guardrail_reasons && review.policy_guardrail_reasons.length > 0) ? (
                          <div className="mt-2 text-xs text-gray-500">
                            {review.policy_guardrail_action ? <span>Guardrail: {review.policy_guardrail_action}</span> : null}
                            {review.policy_guardrail_action && review.policy_guardrail_reasons && review.policy_guardrail_reasons.length > 0 ? <span> • </span> : null}
                            {review.policy_guardrail_reasons && review.policy_guardrail_reasons.length > 0 ? (
                              <span>Reasons: {review.policy_guardrail_reasons.slice(0, 3).join('; ')}</span>
                            ) : null}
                          </div>
                        ) : null}
                        {review.can_approve ||
                        review.can_reject ||
                        review.can_launch_follow_up ||
                        review.can_relaunch_follow_up ||
                        review.can_skip ||
                        review.can_restart ||
                        review.can_resume ||
                        review.can_cancel ||
                        (review.available_actions || []).includes('apply_guardrail') ? (
                          <div className="mt-3">
                            <label className="block text-xs font-medium uppercase tracking-wide text-gray-500">
                              Operator note
                              <textarea
                                className="mt-2 min-h-[72px] w-full rounded-xl border border-gray-300 bg-white px-3 py-2 text-sm text-gray-700"
                                value={reviewNoteDrafts[buildReviewQueueKey(review)] || ''}
                                onChange={(event) =>
                                  setReviewNoteDrafts((prev) => ({
                                    ...prev,
                                    [buildReviewQueueKey(review)]: event.target.value,
                                  }))
                                }
                                placeholder="Optional note for the approval decision"
                              />
                            </label>
                          </div>
                        ) : null}
                        <div className="mt-3 flex flex-wrap gap-2">
                          {review.can_approve ? (
                            <Button
                              size="sm"
                              onClick={() =>
                                actOnReviewMutation.mutate({
                                  review_type: String(review.review_type || ''),
                                  source_kind: String(review.source_kind || ''),
                                  source_id: String(review.source_id || ''),
                                  opportunity_id: String(review.opportunity_id || ''),
                                  action: review.review_type === 'approval_checkpoint' ? 'approve' : 'approve_follow_up',
                                  operator_note: reviewNoteDrafts[buildReviewQueueKey(review)] || undefined,
                                })
                              }
                              disabled={actOnReviewMutation.isLoading}
                            >
                              Approve
                            </Button>
                          ) : null}
                          {review.item_type === 'approval_checkpoint' && (review.available_actions || []).includes('edit') ? (
                            <Button
                              size="sm"
                              onClick={() => {
                                const draft = getCheckpointDraftValue(review);
                                let parsedParams: Record<string, any> = {};
                                try {
                                  parsedParams = draft.params.trim() ? JSON.parse(draft.params) : {};
                                } catch (error: any) {
                                  toast.error(`Invalid JSON params: ${error?.message || 'parse error'}`);
                                  return;
                                }
                                actOnReviewMutation.mutate({
                                  review_type: String(review.review_type || ''),
                                  source_kind: String(review.source_kind || ''),
                                  source_id: String(review.source_id || ''),
                                  opportunity_id: String(review.opportunity_id || ''),
                                  action: 'edit',
                                  operator_note: reviewNoteDrafts[buildReviewQueueKey(review)] || undefined,
                                  checkpoint_action_patch: {
                                    tool: draft.tool.trim() || undefined,
                                    purpose: draft.purpose.trim() || undefined,
                                    params: parsedParams,
                                  },
                                });
                              }}
                              disabled={actOnReviewMutation.isLoading}
                            >
                              Edit + approve
                            </Button>
                          ) : null}
                          {review.can_reject ? (
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() =>
                                actOnReviewMutation.mutate({
                                  review_type: String(review.review_type || ''),
                                  source_kind: String(review.source_kind || ''),
                                  source_id: String(review.source_id || ''),
                                  opportunity_id: String(review.opportunity_id || ''),
                                  action: review.review_type === 'approval_checkpoint' ? 'reject' : 'reject_follow_up',
                                  operator_note: reviewNoteDrafts[buildReviewQueueKey(review)] || undefined,
                                })
                              }
                              disabled={actOnReviewMutation.isLoading}
                            >
                              Reject
                            </Button>
                          ) : null}
                          {review.can_skip ? (
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() =>
                                actOnReviewMutation.mutate({
                                  review_type: String(review.review_type || ''),
                                  source_kind: String(review.source_kind || ''),
                                  source_id: String(review.source_id || ''),
                                  opportunity_id: String(review.opportunity_id || ''),
                                  action: 'skip',
                                  operator_note: reviewNoteDrafts[buildReviewQueueKey(review)] || undefined,
                                })
                              }
                              disabled={actOnReviewMutation.isLoading}
                            >
                              Skip
                            </Button>
                          ) : null}
                          {review.can_launch_follow_up ? (
                            <Button
                              size="sm"
                              onClick={() =>
                                actOnReviewMutation.mutate({
                                  review_type: String(review.review_type || ''),
                                  source_kind: String(review.source_kind || ''),
                                  source_id: String(review.source_id || ''),
                                  opportunity_id: String(review.opportunity_id || ''),
                                  action: 'launch_follow_up',
                                  operator_note: reviewNoteDrafts[buildReviewQueueKey(review)] || undefined,
                                })
                              }
                              disabled={actOnReviewMutation.isLoading}
                            >
                              Launch follow-up
                            </Button>
                          ) : null}
                          {review.can_relaunch_follow_up ? (
                            <Button
                              size="sm"
                              onClick={() =>
                                actOnReviewMutation.mutate({
                                  review_type: String(review.review_type || ''),
                                  source_kind: String(review.source_kind || ''),
                                  source_id: String(review.source_id || ''),
                                  opportunity_id: String(review.opportunity_id || ''),
                                  action: 'relaunch_follow_up',
                                  operator_note: reviewNoteDrafts[buildReviewQueueKey(review)] || undefined,
                                })
                              }
                              disabled={actOnReviewMutation.isLoading}
                            >
                              Relaunch follow-up
                            </Button>
                          ) : null}
                          {review.can_restart ? (
                            <Button
                              size="sm"
                              onClick={() =>
                                actOnReviewMutation.mutate({
                                  review_type: String(review.review_type || ''),
                                  source_kind: String(review.source_kind || ''),
                                  source_id: String(review.source_id || ''),
                                  opportunity_id: String(review.opportunity_id || ''),
                                  action: 'restart',
                                  operator_note: reviewNoteDrafts[buildReviewQueueKey(review)] || undefined,
                                })
                              }
                              disabled={actOnReviewMutation.isLoading}
                            >
                              Restart
                            </Button>
                          ) : null}
                          {review.can_resume ? (
                            <Button
                              size="sm"
                              onClick={() =>
                                actOnReviewMutation.mutate({
                                  review_type: String(review.review_type || ''),
                                  source_kind: String(review.source_kind || ''),
                                  source_id: String(review.source_id || ''),
                                  opportunity_id: String(review.opportunity_id || ''),
                                  action: 'resume',
                                  operator_note: reviewNoteDrafts[buildReviewQueueKey(review)] || undefined,
                                })
                              }
                              disabled={actOnReviewMutation.isLoading}
                            >
                              Resume
                            </Button>
                          ) : null}
                          {review.can_cancel ? (
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() =>
                                actOnReviewMutation.mutate({
                                  review_type: String(review.review_type || ''),
                                  source_kind: String(review.source_kind || ''),
                                  source_id: String(review.source_id || ''),
                                  opportunity_id: String(review.opportunity_id || ''),
                                  action: 'cancel',
                                  operator_note: reviewNoteDrafts[buildReviewQueueKey(review)] || undefined,
                                })
                              }
                              disabled={actOnReviewMutation.isLoading}
                            >
                              Cancel
                            </Button>
                          ) : null}
                          {review.review_type === 'policy_review' && (review.available_actions || []).includes('apply_guardrail') ? (
                            <Button
                              size="sm"
                              onClick={() =>
                                actOnReviewMutation.mutate({
                                  review_type: String(review.review_type || ''),
                                  source_kind: String(review.source_kind || ''),
                                  source_id: String(review.source_id || ''),
                                  opportunity_id: String(review.opportunity_id || ''),
                                  action: 'apply_guardrail',
                                  operator_note: reviewNoteDrafts[buildReviewQueueKey(review)] || undefined,
                                })
                              }
                              disabled={actOnReviewMutation.isLoading}
                            >
                              Apply guardrail
                            </Button>
                          ) : null}
                          {review.review_type === 'policy_review' && review.job_id ? (
                            <Link
                              to={buildHealthMonitorPath(review, { includePolicyHistory: true })}
                              className="rounded-full border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-primary-500 hover:text-gray-900"
                            >
                              Compare before/after
                            </Link>
                          ) : null}
                          {(review.review_type === 'policy_review' || review.review_type === 'budget_review') && review.job_id ? (
                            <Link
                              to={buildHealthMonitorPath(review)}
                              className="rounded-full border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-primary-500 hover:text-gray-900"
                            >
                              Open monitor
                            </Link>
                          ) : null}
                          {(review.review_type === 'policy_review' || review.review_type === 'budget_review') &&
                          review.action_path &&
                          review.source_kind === 'portfolio' ? (
                            <Link
                              to={review.action_path}
                              className="rounded-full border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-primary-500 hover:text-gray-900"
                            >
                              Open fleet
                            </Link>
                          ) : null}
                          {review.queue_path ? (
                            <Link to={review.queue_path} className="rounded-full border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-primary-500 hover:text-gray-900">
                              Open in queue
                            </Link>
                          ) : null}
                          {queueScope === 'global' && review.run_id ? (
                            <button
                              type="button"
                              onClick={() => {
                                const next = new URLSearchParams(searchParams);
                                next.set('run', String(review.run_id));
                                next.delete('node');
                                next.delete('queueScope');
                                next.delete('queueOffset');
                                setSearchParams(next);
                              }}
                              className="rounded-full border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-primary-500 hover:text-gray-900"
                            >
                              Open run
                            </button>
                          ) : null}
                          {review.action_path ? (
                            <Link to={review.action_path} className="rounded-full border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-primary-500 hover:text-gray-900">
                              Open context
                            </Link>
                          ) : null}
                          {review.note_path ? (
                            <Link to={review.note_path} className="rounded-full border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-primary-500 hover:text-gray-900">
                              Open note
                            </Link>
                          ) : null}
                          {review.synthesis_path ? (
                            <Link to={review.synthesis_path} className="rounded-full border border-gray-300 px-3 py-1.5 text-sm font-medium text-gray-700 hover:border-primary-500 hover:text-gray-900">
                              Open synthesis
                            </Link>
                          ) : null}
                        </div>
                      </div>
                    ))}
                    {activeReviewItems.length === 0 ? (
                      <div className="text-sm text-gray-500">
                        {queueScope === 'global'
                          ? 'No queued operator reviews match the active global queue filters.'
                          : detail.queued_operator_reviews.length === 0
                            ? 'No queued operator reviews were linked to this run.'
                            : 'No queued operator reviews match the active queue filters.'}
                      </div>
                    ) : null}
                  </div>
                </div>

                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Routing snapshot</div>
                  <div className="mt-2 rounded-xl bg-gray-100 p-4 text-sm text-gray-700">
                    <div>Summary: {detail.routing?.summary || '—'}</div>
                    <div className="mt-1">Provider/model: {[detail.routing?.provider, detail.routing?.model].filter(Boolean).join(' / ') || '—'}</div>
                    <div className="mt-1">Requests with metadata: {detail.routing?.request_count ?? 0}</div>
                  </div>
                </div>

                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Memory graph</div>
                  <div className="mt-2 rounded-xl bg-gray-100 p-4 text-sm text-gray-700">
                    <div>Nodes: {detail.memory_graph?.stats?.memory_count ?? 0}</div>
                    <div className="mt-1">Edges: {detail.memory_graph?.stats?.edge_count ?? 0}</div>
                    <div className="mt-1">Jobs represented: {detail.memory_graph?.stats?.job_count ?? 0}</div>
                  </div>
                </div>

                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Decision trace</div>
                  <div className="mt-3 space-y-3">
                    {detail.decision_trace.slice(0, 8).map((event) => (
                      <div key={event.event_id} className="rounded-xl border border-gray-200 bg-gray-100 p-3">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium text-gray-900">{event.summary}</div>
                            <div className="mt-1 text-xs text-gray-500">
                              {event.decision_type} • {formatDateTime(event.event_time)}
                            </div>
                          </div>
                          {event.deep_link ? (
                            <Link
                              to={resolveDecisionTraceDeepLink(event.deep_link.target_tab, event.deep_link.params)}
                              className="text-xs font-medium text-gray-700 underline-offset-2 hover:underline"
                            >
                              Open
                            </Link>
                          ) : null}
                        </div>
                      </div>
                    ))}
                    {detail.decision_trace.length === 0 ? <div className="text-sm text-gray-500">No persisted decision events were linked.</div> : null}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-sm text-gray-500">Select a run to see replay, memory, routing, and downstream links.</div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};

export default AgentControlPlanePage;
