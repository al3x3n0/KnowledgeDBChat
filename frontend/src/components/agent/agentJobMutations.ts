/**
 * Job mutations shared by the Runs page and the Research Inbox page.
 *
 * The inbox moved out of Runs to live beside Papers and Reading Lists, but it
 * still creates jobs and actions the follow-up queue. Rather than let the two
 * pages keep their own copies -- the follow-up one is eighty lines, and a fix
 * applied to one copy would not reach the other -- the mutations live here and
 * both pages call them.
 *
 * Everything page-specific is a callback, not a branch: where a launched
 * follow-up should navigate, and what to refetch afterwards, differ per page
 * and are the caller's business. What does NOT differ -- which query keys to
 * invalidate, which toast to show, how the review-row key is cleared on settle
 * -- stays here, because those are properties of the action itself.
 */

import toast from 'react-hot-toast';
import { useMutation, useQueryClient } from 'react-query';

import { apiClient } from '../../services/api';
import { invalidateAgentRunQueries } from '../../utils/agentRunQueries';
import type { AgentJobCreate, AgentJobFromChainCreate } from '../../types';

export interface FollowUpQueueActionVariables {
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
}

export function useCreateAgentJobMutation(
  options: { onCreated?: (job: any) => void; successMessage?: string } = {}
) {
  const queryClient = useQueryClient();
  return useMutation((data: AgentJobCreate) => apiClient.createAgentJob(data), {
    onSuccess: (job) => {
      invalidateAgentRunQueries(queryClient);
      toast.success(options.successMessage || 'Job created');
      options.onCreated?.(job);
    },
    onError: (error: any) => {
      toast.error(error.message || 'Create failed');
    },
  });
}

export function useCreateJobFromChainMutation(options: { onCreated?: (job: any) => void } = {}) {
  const queryClient = useQueryClient();
  return useMutation((data: AgentJobFromChainCreate) => apiClient.createJobFromChain(data), {
    onSuccess: (job) => {
      invalidateAgentRunQueries(queryClient);
      toast.success('Chain started');
      options.onCreated?.(job);
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to start chain');
    },
  });
}

export function useUpsertMonitorProfileMutation() {
  const queryClient = useQueryClient();
  return useMutation(
    (data: {
      customer?: string;
      muted_tokens?: string[];
      muted_patterns?: string[];
      notes?: string;
      merge_lists?: boolean;
    }) => apiClient.upsertResearchMonitorProfile(data),
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
}

export function useFollowUpQueueActionMutation(options: {
  /**
   * Inline review rows track which row is mid-action and clear its note draft
   * afterwards. Only the opportunity surfaces render those rows -- the operator
   * queue actions a follow-up without one -- so both are optional.
   */
  setActiveFollowUpReviewKey?: React.Dispatch<React.SetStateAction<string>>;
  setFollowUpReviewNoteDrafts?: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  /** Refetch whatever list the acted-on row came from. Runs has two; the inbox has none. */
  onRefreshTarget?: (target: 'domain' | 'fleet') => void;
  /** Go to the job a launch produced. Return true if you navigated. */
  onFollowUpLaunched?: (followUpJobId: string) => boolean;
}) {
  const queryClient = useQueryClient();
  const { setActiveFollowUpReviewKey, setFollowUpReviewNoteDrafts } = options;
  return useMutation(
    (variables: FollowUpQueueActionVariables) =>
      apiClient.actionAgentCheckpointQueueFollowUp({
        inbox_item_id: variables.inbox_item_id,
        domain_research_profile_id: variables.domain_research_profile_id,
        profile_opportunity_id: variables.profile_opportunity_id,
        portfolio_id: variables.portfolio_id,
        portfolio_opportunity_id: variables.portfolio_opportunity_id,
        action: variables.action,
        operator_note: variables.operator_note,
      }),
    {
      onMutate: (variables) => {
        if (variables.reviewRowKey) setActiveFollowUpReviewKey?.(variables.reviewRowKey);
      },
      onSuccess: (response, variables) => {
        invalidateAgentRunQueries(queryClient, [
          'research-inbox',
          'research-inbox-stats',
          'research-portfolios',
          'domain-research-profiles',
        ]);
        if (variables.refreshTarget) options.onRefreshTarget?.(variables.refreshTarget);
        if (variables.reviewRowKey) {
          const reviewKey = String(variables.reviewRowKey);
          setFollowUpReviewNoteDrafts?.((prev) => {
            if (!(reviewKey in prev)) return prev;
            const next = { ...prev };
            delete next[reviewKey];
            return next;
          });
        }
        if (response.follow_up_job_id) {
          toast.success('Follow-up launched');
          if (variables.navigateOnLaunch !== false) {
            options.onFollowUpLaunched?.(String(response.follow_up_job_id));
          }
        } else {
          toast.success(response.detail || 'Follow-up decision recorded');
        }
      },
      onError: (error: any) => {
        toast.error(
          error?.response?.data?.detail || error?.message || 'Failed to apply follow-up queue action'
        );
      },
      onSettled: (_data, _error, variables) => {
        if (variables?.reviewRowKey) {
          setActiveFollowUpReviewKey?.((current) =>
            current === variables.reviewRowKey ? '' : current
          );
        }
      },
    }
  );
}
