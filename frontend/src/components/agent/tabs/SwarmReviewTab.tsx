/**
 * Swarm review: the runs whose roles disagreed, and what to do about them.
 *
 * Lifted out of `AutonomousAgentsPage`, which held 169 `useState` for 13 tabs.
 *
 * The six filters and the memo that applies them came with it. They were
 * declared at page level but read by nothing else -- a filter that only this
 * tab sets and only this tab reads is this tab's state, and leaving it on the
 * page made it look shared when it never was.
 */

import { Layers, RefreshCw, XCircle } from 'lucide-react';
import React, { useMemo, useState } from 'react';

import type { AgentJob, CollaborationSummary, User } from '../../../types';
import Button from '../../common/Button';
import LoadingSpinner from '../../common/LoadingSpinner';
import CollaborationSummaryPanel from '../CollaborationSummaryPanel';

export interface SwarmReviewTabProps {
  swarmReviewJobs: AgentJob[];
  /** A query key on the page, not tab state: both swarm queries depend on it. */
  visibilityScope: 'mine' | 'shared' | 'all';
  onVisibilityScopeChange: (scope: 'mine' | 'shared' | 'all') => void;
  swarmReviewJobsLoading: boolean;
  refetchSwarmReviewJobs: () => void;
  swarmAnalyticsData: any;
  swarmAnalyticsLoading: boolean;
  refetchSwarmAnalytics: () => void;
  backlogBySwarmJobId: Record<string, any[]>;
  /** A lookup, not a map: `tsc` caught this typed as a Record. */
  userLabelById: (userId: string) => string;
  collaborationUsers: User[];
  currentUserId?: string;
  noteDrafts: Record<string, string>;
  onNoteDraftsChange: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  actionMutation: any;
  createCodingBacklogMutation: any;
  onOpenJob: (job: AgentJob) => void;
  onGoToBacklog: () => void;
}

export const SwarmReviewTab: React.FC<SwarmReviewTabProps> = ({
  swarmReviewJobs,
  visibilityScope: swarmReviewVisibilityScope,
  onVisibilityScopeChange: setSwarmReviewVisibilityScope,
  swarmReviewJobsLoading,
  refetchSwarmReviewJobs,
  swarmAnalyticsData,
  swarmAnalyticsLoading,
  refetchSwarmAnalytics,
  backlogBySwarmJobId,
  userLabelById,
  collaborationUsers,
  currentUserId,
  noteDrafts: swarmReviewNoteDrafts,
  onNoteDraftsChange: setSwarmReviewNoteDrafts,
  actionMutation,
  createCodingBacklogMutation,
  onOpenJob,
  onGoToBacklog,
}) => {
  const [swarmReviewPresetFilter, setSwarmReviewPresetFilter] = useState('');
  const [swarmReviewStateFilter, setSwarmReviewStateFilter] = useState('');
  const [swarmReviewConfidenceBand, setSwarmReviewConfidenceBand] = useState('');
  const [swarmReviewBacklogFilter, setSwarmReviewBacklogFilter] = useState('');
  const [swarmReviewAssignmentFilter, setSwarmReviewAssignmentFilter] = useState('');

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
      if (swarmReviewAssignmentFilter === 'assigned_to_me' && assignedUserId !== String(currentUserId || '')) return false;
      if (swarmReviewAssignmentFilter === 'unassigned' && assignedUserId) return false;
      if (swarmReviewAssignmentFilter && !['assigned_to_me', 'unassigned'].includes(swarmReviewAssignmentFilter) && assignedUserId !== swarmReviewAssignmentFilter) return false;
      return true;
    });
  }, [swarmReviewJobs, swarmReviewPresetFilter, swarmReviewStateFilter, swarmReviewConfidenceBand, swarmReviewBacklogFilter, swarmReviewAssignmentFilter, backlogBySwarmJobId, currentUserId]);

  return (
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
              <span className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">
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
              <span className="bg-gray-200 text-gray-700 px-2 py-1 rounded">Backlog {Number(row.backlog_handoff_runs || 0)}</span>
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
                      <span className="text-xs px-2 py-1 rounded bg-gray-200 text-gray-700 border border-gray-200">{reviewState.replace(/_/g, ' ')}</span>
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
                    <Button size="sm" variant="ghost" onClick={() => onOpenJob(job)}>
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
                      onClick={() => actionMutation.mutate({ jobId: job.id, action: 'assign_swarm_review', actionPayload: { assigned_user_id: String(currentUserId || '') } })}
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
                      <div key={`${String(candidate.job_id || 'candidate')}-${idx}`} className="border border-gray-200 rounded-lg p-3 bg-gray-100">
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
  );
};

export default SwarmReviewTab;
