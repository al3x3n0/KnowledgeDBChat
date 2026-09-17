import type {
  AnyMutation,
  BuildRunsUrl,
  NavigateFunction,
  QueryClient,
  Refetch,
  SetActiveTab,
} from '../propTypes';
import React, { useCallback, useMemo } from 'react';
import type { QueueHealthDrilldown } from '../drilldowns';
import {
  Activity,
  CheckCircle2,
  RefreshCw,
  XCircle,
} from 'lucide-react';
import type {
  AgentCheckpointQueueItem,
  AgentCheckpointQueueResponse,
  AgentCheckpointQueueAction,
  AgentJobCreate,
  AgentJobFromChainCreate,
} from '../../../types';
import Button from '../../../components/common/Button';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import { apiClient } from '../../../services/api';
import {
  formatQueueHealthDrilldownLabel,
} from '../drilldowns';
import { invalidateAgentRunQueries } from '../../../utils/agentRunQueries';
import {
  summarizeSchedulerState,
} from '../../../utils/agentJobDetail';
import toast from 'react-hot-toast';
import { useMutation } from 'react-query';

export interface OperatorQueueTabProps {
  checkpointQueueData?: AgentCheckpointQueueResponse;
  checkpointQueueLoading: boolean;
  refetchCheckpointQueue: Refetch;
  setActiveTab: SetActiveTab;
  setHealthCustomerFilter: React.Dispatch<React.SetStateAction<string>>;
  setQueueDrafts: React.Dispatch<React.SetStateAction<Record<string, { note: string; showEdit: boolean; tool: string; purpose: string; params: string }>>>;
  setShowInboxMonitorModal: React.Dispatch<React.SetStateAction<boolean>>;
  actionMutation: AnyMutation;
  buildAutonomousAgentsUrl: BuildRunsUrl;
  createFromChainMutation: AnyMutation;
  createMutation: AnyMutation;
  followUpQueueActionMutation: AnyMutation;
  getQueueDraft: (item: AgentCheckpointQueueItem) => { note: string; showEdit: boolean; tool: string; purpose: string; params: string };
  getQueueDraftValue: (item: AgentCheckpointQueueItem) => { note: string; showEdit: boolean; tool: string; purpose: string; params: string };
  navigate: NavigateFunction;
  openHealthPolicyComparison: (monitorJobId: string, historyEntryId?: string) => void;
  openQueueItemTarget: (item: AgentCheckpointQueueItem) => void;
  queryClient: QueryClient;
  queueBulkNote: string;
  setQueueBulkNote: React.Dispatch<React.SetStateAction<string>>;
  queueCustomerFilter: string;
  setQueueCustomerFilter: React.Dispatch<React.SetStateAction<string>>;
  queueEscalationFilter: string;
  setQueueEscalationFilter: React.Dispatch<React.SetStateAction<string>>;
  queueHealthDrilldown: QueueHealthDrilldown;
  setQueueHealthDrilldown: React.Dispatch<React.SetStateAction<QueueHealthDrilldown>>;
  queueItemTypeFilter: string;
  setQueueItemTypeFilter: React.Dispatch<React.SetStateAction<string>>;
  queueJobFilter: string;
  setQueueJobFilter: React.Dispatch<React.SetStateAction<string>>;
  queueJobTypeFilter: string;
  setQueueJobTypeFilter: React.Dispatch<React.SetStateAction<string>>;
  queueOperatorPreset: string;
  setQueueOperatorPreset: React.Dispatch<React.SetStateAction<string>>;
  queueOverdueOnly: boolean;
  setQueueOverdueOnly: React.Dispatch<React.SetStateAction<boolean>>;
  queueSelection: Record<string, boolean>;
  setQueueSelection: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  queueSlaBucketFilter: string;
  setQueueSlaBucketFilter: React.Dispatch<React.SetStateAction<string>>;
  queueSortBy: string;
  setQueueSortBy: React.Dispatch<React.SetStateAction<string>>;
  queueStatusFilter: string;
  setQueueStatusFilter: React.Dispatch<React.SetStateAction<string>>;
  rollbackMonitorPolicyMutation: AnyMutation;
  selectedQueueItems: AgentCheckpointQueueItem[];
  updateMonitorPolicyMutation: AnyMutation;
  visibleQueueItems: AgentCheckpointQueueItem[];
}

export const OperatorQueueTab: React.FC<OperatorQueueTabProps> = ({
  checkpointQueueData,
  checkpointQueueLoading,
  refetchCheckpointQueue,
  setActiveTab,
  setHealthCustomerFilter,
  setQueueDrafts,
  setShowInboxMonitorModal,
  actionMutation,
  buildAutonomousAgentsUrl,
  createFromChainMutation,
  createMutation,
  followUpQueueActionMutation,
  getQueueDraft,
  getQueueDraftValue,
  navigate,
  openHealthPolicyComparison,
  openQueueItemTarget,
  queryClient,
  queueBulkNote,
  setQueueBulkNote,
  queueCustomerFilter,
  setQueueCustomerFilter,
  queueEscalationFilter,
  setQueueEscalationFilter,
  queueHealthDrilldown,
  setQueueHealthDrilldown,
  queueItemTypeFilter,
  setQueueItemTypeFilter,
  queueJobFilter,
  setQueueJobFilter,
  queueJobTypeFilter,
  setQueueJobTypeFilter,
  queueOperatorPreset,
  setQueueOperatorPreset,
  queueOverdueOnly,
  setQueueOverdueOnly,
  queueSelection,
  setQueueSelection,
  queueSlaBucketFilter,
  setQueueSlaBucketFilter,
  queueSortBy,
  setQueueSortBy,
  queueStatusFilter,
  setQueueStatusFilter,
  rollbackMonitorPolicyMutation,
  selectedQueueItems,
  updateMonitorPolicyMutation,
  visibleQueueItems,
}) => {
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

  const toggleQueueSelection = useCallback((item: AgentCheckpointQueueItem) => {
    setQueueSelection((prev) => {
      const next = { ...prev };
      if (next[item.queue_key]) delete next[item.queue_key];
      else next[item.queue_key] = true;
      return next;
    });
  }, [setQueueSelection]);

  const selectVisibleQueueItems = useCallback(() => {
    setQueueSelection((prev) => {
      const next = { ...prev };
      visibleQueueItems.forEach((item) => {
        next[item.queue_key] = true;
      });
      return next;
    });
  }, [visibleQueueItems, setQueueSelection]);

  const clearQueueSelection = useCallback(() => {
    setQueueSelection({});
  }, [setQueueSelection]);

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
  }, [getQueueDraft, setQueueDrafts]);

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
  }, [openHealthPolicyComparison, openQueueItemTarget, rollbackMonitorPolicyMutation, updateMonitorPolicyMutation, setActiveTab, setHealthCustomerFilter]);

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

  return (
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
  );
};

export default OperatorQueueTab;
