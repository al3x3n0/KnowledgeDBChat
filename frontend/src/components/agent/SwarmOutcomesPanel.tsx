/**
 * What the coding swarms actually produced, and how often it was promoted.
 *
 * Presentational on purpose. The filters, the query and the three derived maps
 * live in `useSwarmOutcomes`, because the maps have consumers outside this
 * view -- the backlog reads one and both are handed to the job detail panel --
 * so a panel that owned the query would have quietly emptied them.
 *
 * Navigation arrives as two callbacks rather than the page's
 * `setActiveTab`/`setSelectedJob`/`navigate` trio, so this renders the same
 * wherever it is mounted.
 */

import { RefreshCw, XCircle } from 'lucide-react';
import React from 'react';

import Button from '../common/Button';
import LoadingSpinner from '../common/LoadingSpinner';
import {
  humanizeDecisionTraceValue,
  humanizeSwarmOutcome,
  swarmOutcomeBadgeClass,
} from '../../utils/agentJobDetail';
import type { SwarmOutcomesState } from './useSwarmOutcomes';

export interface SwarmOutcomesPanelProps {
  outcomes: SwarmOutcomesState;
  userLabelById: (userId: string) => string;
  onOpenJob: (jobId: string) => void;
  onOpenBacklog: () => void;
}

const SwarmOutcomesPanel: React.FC<SwarmOutcomesPanelProps> = ({
  outcomes,
  userLabelById,
  onOpenJob,
  onOpenBacklog,
}) => {
  const {
    swarmOutcomePresetFilter,
    setSwarmOutcomePresetFilter,
    swarmOutcomeTerminalFilter,
    setSwarmOutcomeTerminalFilter,
    swarmOutcomePromotionFilter,
    setSwarmOutcomePromotionFilter,
    swarmOutcomeDateRange,
    setSwarmOutcomeDateRange,
    swarmOutcomeVisibilityScope,
    setSwarmOutcomeVisibilityScope,
    swarmOutcomeAnalyticsData,
    swarmOutcomeAnalyticsLoading,
    refetchSwarmOutcomeAnalytics,
    swarmOutcomeCases,
  } = outcomes;

  return (
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
                <span className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">
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
                        <span className="text-xs px-2 py-1 rounded bg-gray-200 text-gray-700 border border-gray-200">
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
                        <div className="mt-1 text-xs text-gray-600">Note: {String(item.review_note)}</div>
                      ) : null}
                      {item.terminal_reason ? (
                        <div className="mt-2 text-xs text-gray-600">{String(item.terminal_reason)}</div>
                      ) : null}
                      {item.review_reason ? (
                        <div className="mt-1 text-xs text-gray-500">{String(item.review_reason)}</div>
                      ) : null}
                    </div>
                    <div className="flex flex-wrap gap-2 shrink-0">
                      <Button size="sm" variant="ghost" onClick={() => onOpenJob(String(item.swarm_job_id))}>
                        Open swarm
                      </Button>
                      {item.repair_job_id ? (
                        <Button size="sm" variant="ghost" onClick={() => onOpenJob(String(item.repair_job_id))}>
                          Open repair
                        </Button>
                      ) : null}
                      {item.backlog_item_id ? (
                        <Button size="sm" variant="ghost" onClick={onOpenBacklog}>
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
  );
};

export default SwarmOutcomesPanel;
