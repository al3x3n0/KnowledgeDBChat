import type {
  AnyMutation,
  BuildRunsUrl,
  NavigateFunction,
  QueryClient,
  Refetch,
  SetActiveTab,
  SetSelectedJob,
} from '../propTypes';
import React, { useCallback, useMemo, useState } from 'react';
import { RefreshCw, XCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import { useMutation } from 'react-query';

import Button from '../../common/Button';
import LoadingSpinner from '../../common/LoadingSpinner';
import CollaborationSummaryPanel from '../CollaborationSummaryPanel';
import { apiClient } from '../../../services/api';
import { copyText } from '../../../utils/clipboard';
import { humanizeSwarmOutcome } from '../../../utils/agentJobDetail';
import type {
  AgentJobSwarmOutcomeCase,
  CodingBacklogItemListResponse,
  User,
  CodingBacklogDecomposition,
  CodingBacklogItem,
  CodingBacklogLatestSummary,
  CodingBacklogPolicy,
  CodingBacklogSlice,
  CodingBacklogTimelineEntry,
  CollaborationSummary,
} from '../../../types';

export interface CodingBacklogTabProps {
  codingBacklogData: CodingBacklogItemListResponse | undefined;
  codingBacklogLoading: boolean;
  refetchCodingBacklog: Refetch;
  setActiveTab: SetActiveTab;
  setSelectedJob: SetSelectedJob;
  swarmOutcomeBySwarmJobId: Record<string, AgentJobSwarmOutcomeCase>;
  user: User | null;
  backlogAssignmentFilter: string;
  setBacklogAssignmentFilter: React.Dispatch<React.SetStateAction<string>>;
  backlogCloseReasonDrafts: Record<string, string>;
  setBacklogCloseReasonDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  backlogCommandsText: string;
  setBacklogCommandsText: React.Dispatch<React.SetStateAction<string>>;
  backlogFailureSymptom: string;
  setBacklogFailureSymptom: React.Dispatch<React.SetStateAction<string>>;
  backlogFilePathsText: string;
  setBacklogFilePathsText: React.Dispatch<React.SetStateAction<string>>;
  backlogGoal: string;
  setBacklogGoal: React.Dispatch<React.SetStateAction<string>>;
  backlogItems: CodingBacklogItem[];
  backlogNoteDrafts: Record<string, string>;
  setBacklogNoteDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  backlogQueueStateFilter: string;
  setBacklogQueueStateFilter: React.Dispatch<React.SetStateAction<string>>;
  backlogSourceId: string;
  setBacklogSourceId: React.Dispatch<React.SetStateAction<string>>;
  backlogTitle: string;
  setBacklogTitle: React.Dispatch<React.SetStateAction<string>>;
  backlogVisibilityScope: 'mine' | 'shared' | 'all';
  setBacklogVisibilityScope: React.Dispatch<React.SetStateAction<'mine' | 'shared' | 'all'>>;
  buildAutonomousAgentsUrl: BuildRunsUrl;
  codeSources: any[];
  collaborationUsers: User[];
  createCodingBacklogMutation: AnyMutation;
  navigate: NavigateFunction;
  queryClient: QueryClient;
  userLabelById: (candidateId?: string | null) => string;
}

export const CodingBacklogTab: React.FC<CodingBacklogTabProps> = ({
  codingBacklogData,
  codingBacklogLoading,
  refetchCodingBacklog,
  setActiveTab,
  setSelectedJob,
  swarmOutcomeBySwarmJobId,
  user,
  backlogAssignmentFilter,
  setBacklogAssignmentFilter,
  backlogCloseReasonDrafts,
  setBacklogCloseReasonDrafts,
  backlogCommandsText,
  setBacklogCommandsText,
  backlogFailureSymptom,
  setBacklogFailureSymptom,
  backlogFilePathsText,
  setBacklogFilePathsText,
  backlogGoal,
  setBacklogGoal,
  backlogItems,
  backlogNoteDrafts,
  setBacklogNoteDrafts,
  backlogQueueStateFilter,
  setBacklogQueueStateFilter,
  backlogSourceId,
  setBacklogSourceId,
  backlogTitle,
  setBacklogTitle,
  backlogVisibilityScope,
  setBacklogVisibilityScope,
  buildAutonomousAgentsUrl,
  codeSources,
  collaborationUsers,
  createCodingBacklogMutation,
  navigate,
  queryClient,
  userLabelById,
}) => {
  const [backlogScope, setBacklogScope] = useState('auto');

  const openBacklogJob = useCallback((jobId: string) => {
    const normalizedJobId = String(jobId || '').trim();
    if (!normalizedJobId) return;
    setActiveTab('jobs');
    navigate(buildAutonomousAgentsUrl(normalizedJobId), { replace: true });
  }, [navigate, buildAutonomousAgentsUrl, setActiveTab]);

  const openPatchPr = useCallback((patchPrId: string) => {
    const normalizedPatchPrId = String(patchPrId || '').trim();
    if (!normalizedPatchPrId) return;
    navigate(`/patch-prs?pr=${encodeURIComponent(normalizedPatchPrId)}`);
  }, [navigate]);

  const downloadBacklogProposal = useCallback(async (proposalId: string, title?: string | null) => {
    const normalizedProposalId = String(proposalId || '').trim();
    if (!normalizedProposalId) return;
    try {
      await apiClient.downloadCodePatchProposal(normalizedProposalId, String(title || `proposal-${normalizedProposalId}`));
    } catch (error: any) {
      toast.error(error?.message || 'Failed to download proposal');
    }
  }, []);

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

  return (
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
                          <span className={`${chipBase} bg-gray-200 text-gray-700`}>{item.status}</span>
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
                            <span className={`${chipBase} ${originatingSwarmRouteMode === 'auto' ? 'bg-amber-50 text-amber-700 border border-amber-100' : 'bg-gray-200 text-gray-700'}`}>
                              {originatingSwarmRouteMode === 'auto' ? 'Auto-routed' : 'Manual backlog'}
                            </span>
                          ) : null}
                          {queueState ? (
                            <span className={`${chipBase} bg-cyan-50 text-cyan-700 border border-cyan-100`}>
                              {queueState.replace(/_/g, ' ')}
                            </span>
                          ) : null}
                          {item.closure_reason ? (
                            <span className={`${chipBase} bg-gray-200 text-gray-700 border border-gray-200`}>
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
                          <div className="text-xs text-gray-600 mt-2">Operator note: {operatorNote}</div>
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
                                      <span className={`${chipBase} ${isActive ? 'bg-violet-100 text-violet-700' : 'bg-gray-200 text-gray-700'}`}>
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
  );
};

export default CodingBacklogTab;
