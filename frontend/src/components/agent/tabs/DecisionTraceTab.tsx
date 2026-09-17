import type {
  AnyMutation,
  BuildRunsUrl,
  NavigateFunction,
  QueryClient,
  Refetch,
  RouterLocation,
  SetActiveTab,
} from '../propTypes';
import React, { useCallback, useMemo } from 'react';
import toast from 'react-hot-toast';
import { useMutation } from 'react-query';
import {
  Download,
  Link2, RefreshCw,
} from 'lucide-react';

import Button from '../../common/Button';
import { apiClient } from '../../../services/api';
import {
  humanizeDecisionTraceValue,
  summarizeSchedulerState,
} from '../../../utils/agentJobDetail';
import type {
  AgentDecisionTraceAnalyticsResponse,
  AgentDecisionTraceResponse,
  AgentDecisionTraceView,
  AgentDecisionTraceEvent,
  User,
} from '../../../types';

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

const decisionTraceSeverityClasses = (value?: string | null) => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'high') return 'bg-rose-100 text-rose-700';
  if (normalized === 'medium') return 'bg-amber-100 text-amber-800';
  return 'bg-gray-200 text-gray-700';
};

const decisionTraceTriageClasses = (value?: string | null) => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'resolved') return 'bg-emerald-100 text-emerald-700';
  if (normalized === 'investigating') return 'bg-blue-100 text-blue-700';
  if (normalized === 'acknowledged') return 'bg-amber-100 text-amber-800';
  return 'bg-rose-100 text-rose-700';
};

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

export interface DecisionTraceTabProps {
  decisionTraceAnalyticsData?: AgentDecisionTraceAnalyticsResponse;
  decisionTraceAnalyticsLoading: boolean;
  decisionTraceData?: AgentDecisionTraceResponse;
  decisionTraceLoading: boolean;
  refetchDecisionTrace: Refetch;
  refetchDecisionTraceAnalytics: Refetch;
  traceViewsData?: { items: AgentDecisionTraceView[] };
  setActiveTab: SetActiveTab;
  applyTraceView: any;
  buildAutonomousAgentsUrl: BuildRunsUrl;
  buildTraceShareUrl: any;
  collaborationUsers: User[];
  currentTraceViewFilters: any;
  decisionTraceActionMutation: AnyMutation;
  expandedTraceEventId: string;
  setExpandedTraceEventId: React.Dispatch<React.SetStateAction<string>>;
  location: RouterLocation;
  navigate: NavigateFunction;
  queryClient: QueryClient;
  selectedTraceViewId: string;
  setSelectedTraceViewId: React.Dispatch<React.SetStateAction<string>>;
  traceActionNoteDrafts: Record<string, string>;
  setTraceActionNoteDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  traceActionableOnly: boolean;
  setTraceActionableOnly: React.Dispatch<React.SetStateAction<boolean>>;
  traceActorModeFilter: string;
  setTraceActorModeFilter: React.Dispatch<React.SetStateAction<string>>;
  traceAssignedToUserIdFilter: string;
  setTraceAssignedToUserIdFilter: React.Dispatch<React.SetStateAction<string>>;
  traceAssigneeDrafts: Record<string, string>;
  setTraceAssigneeDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  traceCustomerFilter: string;
  setTraceCustomerFilter: React.Dispatch<React.SetStateAction<string>>;
  traceDateRange: string;
  setTraceDateRange: React.Dispatch<React.SetStateAction<string>>;
  traceDecisionTypeFilter: string;
  setTraceDecisionTypeFilter: React.Dispatch<React.SetStateAction<string>>;
  traceDueAtDrafts: Record<string, string>;
  setTraceDueAtDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  traceEscalationStateFilter: string;
  setTraceEscalationStateFilter: React.Dispatch<React.SetStateAction<string>>;
  traceFiltersDirtyRef: any;
  traceOffset: number;
  setTraceOffset: React.Dispatch<React.SetStateAction<number>>;
  traceOperatorPreset: string;
  setTraceOperatorPreset: React.Dispatch<React.SetStateAction<string>>;
  tracePinnedOnly: boolean;
  setTracePinnedOnly: React.Dispatch<React.SetStateAction<boolean>>;
  traceSeverityFilter: string;
  setTraceSeverityFilter: React.Dispatch<React.SetStateAction<string>>;
  traceSourceKindFilter: string;
  setTraceSourceKindFilter: React.Dispatch<React.SetStateAction<string>>;
  traceStartAt: any;
  traceStatusFilter: string;
  setTraceStatusFilter: React.Dispatch<React.SetStateAction<string>>;
  traceTriageStatusFilter: string;
  setTraceTriageStatusFilter: React.Dispatch<React.SetStateAction<string>>;
  traceUnassignedOnly: boolean;
  setTraceUnassignedOnly: React.Dispatch<React.SetStateAction<boolean>>;
  traceViewIsDefaultDraft: boolean;
  setTraceViewIsDefaultDraft: React.Dispatch<React.SetStateAction<boolean>>;
  traceViewNameDraft: string;
  setTraceViewNameDraft: React.Dispatch<React.SetStateAction<string>>;
  userLabelById: any;
}

export const DecisionTraceTab: React.FC<DecisionTraceTabProps> = ({
  decisionTraceAnalyticsData,
  decisionTraceAnalyticsLoading,
  decisionTraceData,
  decisionTraceLoading,
  refetchDecisionTrace,
  refetchDecisionTraceAnalytics,
  traceViewsData,
  setActiveTab,
  applyTraceView,
  buildAutonomousAgentsUrl,
  buildTraceShareUrl,
  collaborationUsers,
  currentTraceViewFilters,
  decisionTraceActionMutation,
  expandedTraceEventId,
  setExpandedTraceEventId,
  location,
  navigate,
  queryClient,
  selectedTraceViewId,
  setSelectedTraceViewId,
  traceActionNoteDrafts,
  setTraceActionNoteDrafts,
  traceActionableOnly,
  setTraceActionableOnly,
  traceActorModeFilter,
  setTraceActorModeFilter,
  traceAssignedToUserIdFilter,
  setTraceAssignedToUserIdFilter,
  traceAssigneeDrafts,
  setTraceAssigneeDrafts,
  traceCustomerFilter,
  setTraceCustomerFilter,
  traceDateRange,
  setTraceDateRange,
  traceDecisionTypeFilter,
  setTraceDecisionTypeFilter,
  traceDueAtDrafts,
  setTraceDueAtDrafts,
  traceEscalationStateFilter,
  setTraceEscalationStateFilter,
  traceFiltersDirtyRef,
  traceOffset,
  setTraceOffset,
  traceOperatorPreset,
  setTraceOperatorPreset,
  tracePinnedOnly,
  setTracePinnedOnly,
  traceSeverityFilter,
  setTraceSeverityFilter,
  traceSourceKindFilter,
  setTraceSourceKindFilter,
  traceStartAt,
  traceStatusFilter,
  setTraceStatusFilter,
  traceTriageStatusFilter,
  setTraceTriageStatusFilter,
  traceUnassignedOnly,
  setTraceUnassignedOnly,
  traceViewIsDefaultDraft,
  setTraceViewIsDefaultDraft,
  traceViewNameDraft,
  setTraceViewNameDraft,
  userLabelById,
}) => {
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
    } else if (
      deepLink.target_tab === 'domain'
      || deepLink.target_tab === 'fleet'
      || deepLink.target_tab === 'inbox'
    ) {
      // These three left the Runs page for destinations of their own. A trace
      // event still carries the old target_tab, so translate it here rather
      // than switching to a tab that no longer exists -- which is what this
      // did until the prop was typed.
      const carry = new URLSearchParams();
      Object.entries(deepLink.params || {}).forEach(([key, value]) => {
        const text = String(value ?? '').trim();
        if (text && key !== 'tab') carry.set(key, text);
      });
      const query = carry.toString();
      const base = deepLink.target_tab === 'domain'
        ? '/settings/domain-profiles'
        : deepLink.target_tab === 'fleet'
          ? '/research/fleet'
          : '/research/inbox';
      navigate(query ? `${base}?${query}` : base);
      return;
    } else if (deepLink.target_tab === 'jobs') {
      setActiveTab('jobs');
    }
    navigate(buildAutonomousAgentsUrl(nextJobId, params), { replace: true });
  }, [buildAutonomousAgentsUrl, navigate, setActiveTab]);

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

  const markTraceFiltersDirty = useCallback(() => {
    traceFiltersDirtyRef.current = true;
  }, [traceFiltersDirtyRef]);

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

  return (
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
                  <span key={value} className="inline-flex items-center rounded-full bg-gray-200 px-2 py-1 text-xs text-gray-700">
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
  );
};

export default DecisionTraceTab;
