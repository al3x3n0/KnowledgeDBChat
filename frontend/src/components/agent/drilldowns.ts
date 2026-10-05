/**
 * Drilldown parameters shared by the page and its tabs.
 *
 * These three pairs travel together: the page normalises a URL parameter into
 * one of the unions, and whichever tab renders that drilldown formats it back
 * into a label for the "showing X" banner. Splitting them across a page and two
 * tab components would put the parser and its printer in different files, where
 * adding a variant to one and not the other type-checks.
 */

export type InboxHealthDrilldown = '' | 'completed_follow_up' | 'failed_follow_up' | 'cancelled_follow_up' | 'suppressed_relaunch';
export type InboxPolicyDrilldown = '' | 'simulated_policy_impact' | 'policy_evaluation_after_rollout';
export type QueueHealthDrilldown = '' | 'pending_follow_up_approvals' | 'manual_follow_up_recommendations' | 'blocked_follow_up';

export const normalizeInboxHealthDrilldown = (value: unknown): InboxHealthDrilldown => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'completed_follow_up') return 'completed_follow_up';
  if (normalized === 'failed_follow_up') return 'failed_follow_up';
  if (normalized === 'cancelled_follow_up') return 'cancelled_follow_up';
  if (normalized === 'suppressed_relaunch') return 'suppressed_relaunch';
  return '';
};

export const normalizeInboxPolicyDrilldown = (value: unknown): InboxPolicyDrilldown => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'simulated_policy_impact') return 'simulated_policy_impact';
  if (normalized === 'policy_evaluation_after_rollout') return 'policy_evaluation_after_rollout';
  return '';
};

export const normalizeQueueHealthDrilldown = (value: unknown): QueueHealthDrilldown => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'pending_follow_up_approvals') return 'pending_follow_up_approvals';
  if (normalized === 'manual_follow_up_recommendations') return 'manual_follow_up_recommendations';
  if (normalized === 'blocked_follow_up') return 'blocked_follow_up';
  return '';
};

export const formatInboxHealthDrilldownLabel = (value: InboxHealthDrilldown): string => {
  if (value === 'completed_follow_up') return 'completed outcomes';
  if (value === 'failed_follow_up') return 'failed outcomes';
  if (value === 'cancelled_follow_up') return 'cancelled outcomes';
  if (value === 'suppressed_relaunch') return 'suppressed relaunches';
  return '';
};

export const formatInboxPolicyDrilldownLabel = (value: InboxPolicyDrilldown): string => {
  if (value === 'simulated_policy_impact') return 'simulated policy impact';
  if (value === 'policy_evaluation_after_rollout') return 'post-rollout evaluation';
  return '';
};

export const formatQueueHealthDrilldownLabel = (value: QueueHealthDrilldown): string => {
  if (value === 'pending_follow_up_approvals') return 'pending approvals';
  if (value === 'manual_follow_up_recommendations') return 'manual recommendations';
  if (value === 'blocked_follow_up') return 'blocked follow-ups';
  return '';
};


/** Every filter is a URL parameter, so a filtered inbox is a link. */
const FILTER_PARAMS = [
  'inbox_status', 'inbox_type', 'inbox_q', 'inbox_job',
  'inbox_customer', 'inbox_health_drilldown', 'inbox_policy_drilldown',
  // Not a filter: focuses one item. The tab reads it straight off location.
  'inbox',
] as const;

export type InboxUrlParams = typeof FILTER_PARAMS[number];

export function buildResearchInboxUrl(params: Partial<Record<InboxUrlParams, string | null>> = {}) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    const normalized = String(value ?? '').trim();
    if (normalized) search.set(key, normalized);
  });
  const query = search.toString();
  return query ? `/research/inbox?${query}` : '/research/inbox';
}

