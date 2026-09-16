/**
 * Autonomy health: what the monitors are doing, and whether their policies hold.
 *
 * The largest of the thirteen tabs at 1,719 lines.
 *
 * What it does NOT own is the drilldowns. "Open the inbox filtered to this
 * customer" touches seven inbox filters and the active tab; "open the queue"
 * touches four more. Those helpers are only ever called from here, so a
 * dependency scan calls them movable -- but moving them would drag fourteen
 * page setters along. Movable is not the same as should-move. They stay on the
 * page and arrive as three callbacks.
 *
 * `healthCustomers` also stays: it is derived from the same response and fills
 * the *inbox* tab's customer filter too. That sharing was a bug when the query
 * behind it was gated to this tab alone, and is now deliberate.
 */

import {
  Activity,
  Brain,
  RefreshCw,
  XCircle,
} from 'lucide-react';
import toast from 'react-hot-toast';
import React, { useCallback, useMemo, useRef, useState } from 'react';
import { useMutation, useQueryClient } from 'react-query';

import { apiClient } from '../../../services/api';
import type {
  ResearchMonitorAnalyticsResponse,
  ResearchMonitorCustomerPortfolio,
  ResearchMonitorCustomerRebalanceEvaluationDetail,
  ResearchMonitorCustomerRebalancePreview,
  ResearchMonitorHealthSummary,
  ResearchMonitorPolicyEvaluationDetail,
  ResearchMonitorPolicyHistoryEntry,
  ResearchMonitorPolicySimulationResponse,
} from '../../../types';
import { invalidateAgentRunQueries } from '../../../utils/agentRunQueries';
import Button from '../../common/Button';
import LoadingSpinner from '../../common/LoadingSpinner';



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

const HEALTH_FOCUS_CARD_CLASS = 'border-cyan-300 bg-cyan-50 ring-2 ring-cyan-200';

/**
 * What the page's `filteredMonitorAnalytics` memo actually produces.
 *
 * Not `ResearchMonitorAnalyticsResponse`: the memo narrows it, dropping
 * `generated_at` and `customers`. Typing this prop properly is what surfaced
 * that -- as `any` it read like the API response and was not one.
 */
export interface FilteredMonitorAnalytics {
  monitors: ResearchMonitorHealthSummary[];
  totals: ResearchMonitorAnalyticsResponse['totals'];
  recommendations: ResearchMonitorAnalyticsResponse['recommendations'];
}

type HealthPolicyDraft = { automation_profile: string; mode: string; allowed: string[] };

export interface AutonomyHealthTabProps {
  onDrillIntoInbox: (drilldown: any, context?: { customer?: string | null; monitorJobId?: string | null }) => void;
  onDrillIntoQueue: (drilldown: any, context?: { customer?: string | null; monitorJobId?: string | null }) => void;
  onOpenInboxForMonitorSignal: (monitorJobId: string, inboxItemId?: string, policyDrilldown?: any) => void;
  /** The *router's* location. Without this the moved body resolved bare
   *  `location` to the DOM global -- which type-checks, has `.search` and
   *  `.pathname`, and is not the router's state. Only CRA's
   *  `no-restricted-globals` caught it; tsc and all 129 tests did not. */
  healthPolicyDrafts: Record<string, HealthPolicyDraft>;
  setHealthPolicyDrafts: React.Dispatch<React.SetStateAction<Record<string, HealthPolicyDraft>>>;
  location: { pathname: string; search: string };
  monitorAnalyticsLoading: boolean;
  refetchMonitorAnalytics: () => void;
  setHealthAutonomyFilter: any;
  setHealthBucketFilter: any;
  setHealthMonitorTypeFilter: any;
  healthCustomerRebalanceEvaluations: Record<string, ResearchMonitorCustomerRebalanceEvaluationDetail>;
  setHealthCustomerRebalanceEvaluations: React.Dispatch<React.SetStateAction<Record<string, ResearchMonitorCustomerRebalanceEvaluationDetail>>>;
  healthPolicySimulations: Record<string, ResearchMonitorPolicySimulationResponse>;
  setHealthPolicySimulations: React.Dispatch<React.SetStateAction<Record<string, ResearchMonitorPolicySimulationResponse>>>;
  healthPolicyEvaluations: Record<string, ResearchMonitorPolicyEvaluationDetail>;
  setHealthPolicyEvaluations: React.Dispatch<React.SetStateAction<Record<string, ResearchMonitorPolicyEvaluationDetail>>>;
  formatAutonomyLabel: any;
  formatReviewModeLabel: any;
  healthAutonomyFilter: any;
  healthBucketFilter: any;
  healthMonitorTypeFilter: any;
  loadPolicyEvaluationMutation: any;
  rollbackMonitorPolicyMutation: any;
  updateMonitorPolicyMutation: any;
  monitorAnalyticsData: any;


  // Kept on the page and passed in. The seven inbox setters and three
  // queue setters below are called directly from this body in seventeen
  // places -- each of them a hand-rolled variant of the drilldown helpers
  // above. Collapsing them would change behaviour, which a refactor
  // should not do quietly, so they are threaded through as they are.
  buildAutonomousAgentsUrl: any;
  deepLinkedHealthMonitor: any;
  filteredMonitorAnalytics: FilteredMonitorAnalytics;
  healthCustomerFilter: string;
  healthCustomers: string[];
  navigate: any;
  openHealthPolicyComparison: any;
  setActiveTab: any;
  setHealthCustomerFilter: React.Dispatch<React.SetStateAction<string>>;
  setInboxCustomerFilter: any;
  setInboxHealthDrilldown: any;
  setInboxJobFilter: any;
  setInboxPolicyDrilldown: any;
  setInboxSearch: any;
  setInboxStatusFilter: any;
  setInboxTypeFilter: any;
  setQueueCustomerFilter: any;
  setQueueHealthDrilldown: any;
  setQueueJobFilter: any;
  setShowMonitorProfilesModal: any;
}

export const AutonomyHealthTab: React.FC<AutonomyHealthTabProps> = ({
  onDrillIntoInbox: openInboxHealthDrilldown,
  onDrillIntoQueue: openQueueHealthDrilldown,
  onOpenInboxForMonitorSignal: openInboxForMonitorSignal,
  healthPolicyDrafts,
  setHealthPolicyDrafts,
  location,
  monitorAnalyticsLoading,
  refetchMonitorAnalytics,
  setHealthAutonomyFilter,
  setHealthBucketFilter,
  setHealthMonitorTypeFilter,
  healthCustomerRebalanceEvaluations,
  setHealthCustomerRebalanceEvaluations,
  healthPolicySimulations,
  setHealthPolicySimulations,
  healthPolicyEvaluations,
  setHealthPolicyEvaluations,
  formatAutonomyLabel,
  formatReviewModeLabel,
  healthAutonomyFilter,
  healthBucketFilter,
  healthMonitorTypeFilter,
  loadPolicyEvaluationMutation,
  rollbackMonitorPolicyMutation,
  updateMonitorPolicyMutation,
  monitorAnalyticsData,

  buildAutonomousAgentsUrl,
  deepLinkedHealthMonitor,
  filteredMonitorAnalytics,
  healthCustomerFilter,
  healthCustomers,
  navigate,
  openHealthPolicyComparison,
  setActiveTab,
  setHealthCustomerFilter,
  setInboxCustomerFilter,
  setInboxHealthDrilldown,
  setInboxJobFilter,
  setInboxPolicyDrilldown,
  setInboxSearch,
  setInboxStatusFilter,
  setInboxTypeFilter,
  setQueueCustomerFilter,
  setQueueHealthDrilldown,
  setQueueJobFilter,
  setShowMonitorProfilesModal,
}) => {
  const queryClient = useQueryClient();
  const healthMonitorCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [healthBudgetDrafts, setHealthBudgetDrafts] = useState<Record<string, { auto_launch_limit_24h: number; approval_queue_limit_24h: number; alert_limit_24h: number; queue_backlog_cap: number }>>({});
  const [healthCustomerBudgetDrafts, setHealthCustomerBudgetDrafts] = useState<Record<string, { auto_launch_limit_24h: number; approval_queue_limit_24h: number; alert_limit_24h: number; queue_backlog_cap: number }>>({});
  const [healthCustomerRebalancePreviews, setHealthCustomerRebalancePreviews] = useState<Record<string, ResearchMonitorCustomerRebalancePreview>>({});

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
  }, [setActiveTab, setHealthCustomerFilter, buildAutonomousAgentsUrl, navigate]);


  const customerPortfolioRows = useMemo(
    () => ((monitorAnalyticsData as ResearchMonitorAnalyticsResponse | undefined)?.customers || []),
    [monitorAnalyticsData]
  );

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
  }, [setHealthPolicyDrafts]);

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
        invalidateAgentRunQueries(queryClient, [
          'research-monitor-analytics',
        ]);
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
        invalidateAgentRunQueries(queryClient, [
          'research-monitor-analytics',
          'research-inbox',
        ]);
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
        invalidateAgentRunQueries(queryClient, [
          'research-monitor-analytics',
          'research-inbox',
        ]);
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

  return (
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
                        <span className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">
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
                        <span className="bg-gray-200 text-gray-700 px-2 py-1 rounded">
                          Auto {customerRow.auto_launch_used_24h}/{customerRow.auto_launch_capacity_24h}
                        </span>
                        <span className="bg-gray-200 text-gray-700 px-2 py-1 rounded">
                          Queue {customerRow.approval_queue_used_24h}/{customerRow.approval_queue_capacity_24h}
                        </span>
                        <span className="bg-gray-200 text-gray-700 px-2 py-1 rounded">
                          Alerts {customerRow.alert_used_24h}/{customerRow.alert_capacity_24h}
                        </span>
                        <span className="bg-gray-200 text-gray-700 px-2 py-1 rounded">
                          Backlog {customerRow.backlog_used}/{customerRow.backlog_capacity}
                        </span>
                        <span className="bg-gray-200 text-gray-700 px-2 py-1 rounded">
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
                                  : 'bg-gray-200 text-gray-700'
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
                  <div className="mt-3 border border-gray-200 rounded p-3 bg-gray-100">
                    <div className="flex items-center justify-between gap-2 mb-3">
                      <div>
                        <div className="text-xs font-medium text-gray-700">Shared customer budget</div>
                        <div className="text-[11px] text-gray-500">
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
                          <div className="text-[11px] text-gray-600 mb-1">{label}</div>
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
                        <div className="mt-3 rounded border border-gray-200 bg-white p-3 text-xs">
                          <div className="font-medium text-gray-800">Rebalance preview</div>
                          <div className="mt-1 text-gray-600">
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
                          <div className="text-gray-600">
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
                              <div key={`${rebalancePreview.customer}-preview-${change.monitor_job_id}`} className="rounded bg-gray-100 px-2 py-2">
                                <div className="font-medium text-gray-800">{change.monitor_name}</div>
                                <div className="text-gray-600 mt-1">
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
                                              : 'bg-gray-200 text-gray-700'
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
                                <div className="mt-3 border border-gray-200 rounded bg-white p-3">
                                  <div className="flex items-center gap-2 flex-wrap">
                                    <span className={`text-[11px] px-2 py-1 rounded ${
                                      evaluationDetail.evaluation_status === 'improving'
                                        ? 'bg-emerald-100 text-emerald-700'
                                        : evaluationDetail.evaluation_status === 'degrading'
                                          ? 'bg-rose-100 text-rose-700'
                                          : evaluationDetail.evaluation_status === 'mixed'
                                            ? 'bg-amber-100 text-amber-800'
                                            : 'bg-gray-200 text-gray-700'
                                    }`}>
                                      {formatPolicyEvaluationStatus(evaluationDetail.evaluation_status)}
                                    </span>
                                    <span className="text-[11px] text-gray-600">
                                      {evaluationDetail.evaluation_sample_count}/{evaluationDetail.evaluation_target_count} accepted signals after rebalance
                                    </span>
                                  </div>
                                  <div className="grid grid-cols-3 gap-3 mt-3 text-[11px]">
                                    <div className="border border-gray-200 rounded p-2">
                                      <div className="font-medium text-gray-700">Before</div>
                                      <div className="mt-1 text-gray-600">
                                        Backlog {evaluationDetail.before_counts.backlog_used} · Throttled {evaluationDetail.before_counts.throttled_monitor_count} · Blocked {evaluationDetail.before_counts.blocked_count}
                                      </div>
                                    </div>
                                    <div className="border border-gray-200 rounded p-2">
                                      <div className="font-medium text-gray-700">After</div>
                                      <div className="mt-1 text-gray-600">
                                        Backlog {evaluationDetail.after_counts.backlog_used} · Throttled {evaluationDetail.after_counts.throttled_monitor_count} · Blocked {evaluationDetail.after_counts.blocked_count}
                                      </div>
                                    </div>
                                    <div className="border border-gray-200 rounded p-2">
                                      <div className="font-medium text-gray-700">Delta</div>
                                      <div className="mt-1 text-gray-600">
                                        Backlog {formatSimulationCountDelta(evaluationDetail.delta_counts.backlog_used)} · Throttled {formatSimulationCountDelta(evaluationDetail.delta_counts.throttled_monitor_count)} · Blocked {formatSimulationCountDelta(evaluationDetail.delta_counts.blocked_count)}
                                      </div>
                                    </div>
                                  </div>
                                  {(evaluationDetail.evaluation_reasons || []).length > 0 ? (
                                    <div className="mt-3 flex flex-wrap gap-2">
                                      {evaluationDetail.evaluation_reasons.map((reason) => (
                                        <span key={reason} className="text-[11px] bg-gray-100 text-gray-700 border border-gray-200 px-2 py-1 rounded">
                                          {reason}
                                        </span>
                                      ))}
                                    </div>
                                  ) : null}
                                  {(evaluationDetail.sample_items || []).length > 0 ? (
                                    <div className="mt-3 space-y-2">
                                      <div className="text-[11px] font-medium text-gray-700">Sample signals</div>
                                      {evaluationDetail.sample_items.map((sample) => (
                                        <div key={`${sample.period}-${sample.item_id}`} className="border border-gray-200 rounded p-2">
                                          <div className="flex items-start justify-between gap-3">
                                            <div className="min-w-0">
                                              <div className="text-xs font-medium text-gray-900">{sample.title}</div>
                                              <div className="text-[11px] text-gray-600 mt-1">
                                                {sample.period} · {sample.monitor_name || 'Unknown monitor'}
                                                {sample.launch_status ? ` · ${sample.launch_status.replace(/_/g, ' ')}` : ''}
                                                {sample.outcome_status ? ` · ${sample.outcome_status.replace(/_/g, ' ')}` : ''}
                                              </div>
                                              {sample.summary ? (
                                                <div className="text-[11px] text-gray-600 mt-1">{sample.summary}</div>
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
                    <div className="bg-gray-100 rounded p-3">
                      <div className="font-medium text-gray-700 mb-1">Top launch monitors</div>
                      {(customerRow.top_launch_monitors || []).length > 0 ? (
                        <div className="space-y-1">
                          {customerRow.top_launch_monitors.map((row) => (
                            <div key={`${customerRow.customer}-launch-${row.monitor_name}`} className="text-gray-600">
                              {row.monitor_name} · {row.value}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-gray-500">No recent launch pressure.</div>
                      )}
                    </div>
                    <div className="bg-gray-100 rounded p-3">
                      <div className="font-medium text-gray-700 mb-1">Top backlog / alerts</div>
                      <div className="space-y-1">
                        {(customerRow.top_backlog_monitors || []).slice(0, 2).map((row) => (
                          <div key={`${customerRow.customer}-backlog-${row.monitor_name}`} className="text-gray-600">
                            Backlog: {row.monitor_name} · {row.value}
                          </div>
                        ))}
                        {(customerRow.top_alert_monitors || []).slice(0, 1).map((row) => (
                          <div key={`${customerRow.customer}-alert-${row.monitor_name}`} className="text-gray-600">
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
                        <span className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">{monitor.monitor_job_type}</span>
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
                                  : 'bg-gray-200 text-gray-700'
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
                      <span className="text-[11px] bg-gray-200 text-gray-700 px-2 py-1 rounded">
                        Auto {monitor.budget_usage?.auto_launch_count_24h || 0}/{monitor.autonomy_budget?.auto_launch_limit_24h || 0}
                      </span>
                      <span className="text-[11px] bg-gray-200 text-gray-700 px-2 py-1 rounded">
                        Queue {monitor.budget_usage?.approval_queue_count_24h || 0}/{monitor.autonomy_budget?.approval_queue_limit_24h || 0}
                      </span>
                      <span className="text-[11px] bg-gray-200 text-gray-700 px-2 py-1 rounded">
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
                        <span className="text-[11px] bg-gray-200 text-gray-700 px-2 py-1 rounded">
                          Post-change sample {monitor.latest_policy_evaluation_sample_count}/{monitor.latest_policy_evaluation_target_count || monitor.latest_policy_evaluation_sample_count}
                        </span>
                        {(monitor.latest_policy_evaluation_reasons || []).map((reason) => (
                          <span key={reason} className="text-[11px] bg-gray-100 text-gray-700 px-2 py-1 rounded border border-gray-200">
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
                  <div className="border border-gray-200 rounded p-3 bg-gray-100">
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
                          <div className="mt-2 text-gray-900">
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
                                <div className="text-sm font-medium text-gray-900">{sample.title}</div>
                                <div className="text-[11px] text-gray-600 mt-1">
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
                                                : 'bg-gray-200 text-gray-700'
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
                                <div className="mt-3 border border-gray-200 rounded bg-white p-3">
                                  <div className="flex items-center gap-2 flex-wrap">
                                    <span
                                      className={`text-[11px] px-2 py-1 rounded ${
                                        evaluationDetail.evaluation_status === 'improving'
                                          ? 'bg-emerald-100 text-emerald-700'
                                          : evaluationDetail.evaluation_status === 'degrading'
                                            ? 'bg-rose-100 text-rose-700'
                                            : evaluationDetail.evaluation_status === 'mixed'
                                              ? 'bg-amber-100 text-amber-800'
                                              : 'bg-gray-200 text-gray-700'
                                      }`}
                                    >
                                      {formatPolicyEvaluationStatus(evaluationDetail.evaluation_status)}
                                    </span>
                                    <span className="text-[11px] text-gray-600">
                                      {evaluationDetail.evaluation_sample_count}/{evaluationDetail.evaluation_target_count} accepted signals after rollout
                                    </span>
                                  </div>
                                  <div className="grid grid-cols-3 gap-3 mt-3 text-[11px]">
                                    <div className="border border-gray-200 rounded p-2">
                                      <div className="font-medium text-gray-700">Before</div>
                                      <div className="mt-1 text-gray-600">
                                        Completed {evaluationDetail.before_counts.follow_up_completed_count} · Failed {evaluationDetail.before_counts.follow_up_failed_count} · Blocked {evaluationDetail.before_counts.blocked_count}
                                      </div>
                                    </div>
                                    <div className="border border-gray-200 rounded p-2">
                                      <div className="font-medium text-gray-700">After</div>
                                      <div className="mt-1 text-gray-600">
                                        Completed {evaluationDetail.after_counts.follow_up_completed_count} · Failed {evaluationDetail.after_counts.follow_up_failed_count} · Blocked {evaluationDetail.after_counts.blocked_count}
                                      </div>
                                    </div>
                                    <div className="border border-gray-200 rounded p-2">
                                      <div className="font-medium text-gray-700">Delta</div>
                                      <div className="mt-1 text-gray-600">
                                        Completed {formatSimulationCountDelta(evaluationDetail.delta_counts.follow_up_completed_count)} · Failed {formatSimulationCountDelta(evaluationDetail.delta_counts.follow_up_failed_count)} · Blocked {formatSimulationCountDelta(evaluationDetail.delta_counts.blocked_count)}
                                      </div>
                                    </div>
                                  </div>
                                  {(evaluationDetail.evaluation_reasons || []).length > 0 ? (
                                    <div className="mt-3 flex flex-wrap gap-2">
                                      {evaluationDetail.evaluation_reasons.map((reason) => (
                                        <span key={reason} className="text-[11px] bg-gray-100 text-gray-700 border border-gray-200 px-2 py-1 rounded">
                                          {reason}
                                        </span>
                                      ))}
                                    </div>
                                  ) : null}
                                  {(evaluationDetail.sample_items || []).length > 0 ? (
                                    <div className="mt-3 space-y-2">
                                      <div className="text-[11px] font-medium text-gray-700">Sample signals</div>
                                      {evaluationDetail.sample_items.map((sample) => (
                                        <div key={`${sample.period}-${sample.item_id}`} className="border border-gray-200 rounded p-2">
                                          <div className="flex items-start justify-between gap-3">
                                            <div className="min-w-0">
                                              <div className="text-xs font-medium text-gray-900">{sample.title}</div>
                                              <div className="text-[11px] text-gray-600 mt-1">
                                                {sample.period} · {sample.launch_status ? sample.launch_status.replace(/_/g, ' ') : 'no launch'}
                                                {sample.outcome_status ? ` · ${sample.outcome_status.replace(/_/g, ' ')}` : ''}
                                                {sample.recommendation_key ? ` · ${sample.recommendation_key}` : ''}
                                              </div>
                                              {sample.summary ? (
                                                <div className="text-[11px] text-gray-600 mt-1">{sample.summary}</div>
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
                        <span key={recommendation.recommendation_key} className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">
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
  );
};

export default AutonomyHealthTab;
