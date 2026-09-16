/**
 * Every surface that shows a run, invalidated together when one changes.
 *
 * The Runs page's Checkpoints tab and the Control Plane's operator review
 * queue are projections of the same backend function --
 * `_build_checkpoint_queue_items`, which the control plane filters to
 * `approval_checkpoint` and `job_recovery` and groups per run.
 *
 * They did not tell each other. Approving from Runs invalidated
 * `agent-checkpoint-queue` and not `agent-control-reviews`; approving from the
 * Control Plane did the reverse. An item actioned in one surface went on being
 * offered in the other until something unrelated happened to refetch it.
 *
 * Twenty-four mutations on the Runs page invalidate the checkpoint queue --
 * not only approvals, but creating, deleting and re-prioritising a job. All of
 * them change what the Control Plane shows too, so the rule is not "a review
 * happened" but "a run changed", and it lives in one place.
 *
 * That is the point of the list being here rather than in each page: the
 * failure was never a mistyped key, it was that touching a run in one surface
 * requires knowing every other surface that shows runs.
 */

import type { QueryClient } from 'react-query';

/** Everything that can be stale once a run changes. */
export const AGENT_RUN_QUERY_KEYS = [
  // Runs page
  'agent-jobs',
  'agent-jobs-stats',
  'agent-checkpoint-queue',
  // Control Plane
  'agent-control-runs',
  'agent-control-run',
  'agent-control-reviews',
] as const;

/**
 * Invalidate every surface showing the run that changed.
 *
 * `extra` carries keys only one caller needs -- the Control Plane also
 * refreshes portfolios and domain profiles, which the Runs queue has no reason
 * to touch.
 */
export function invalidateAgentRunQueries(
  queryClient: QueryClient,
  extra: readonly string[] = []
): void {
  [...AGENT_RUN_QUERY_KEYS, ...extra].forEach((key) => {
    queryClient.invalidateQueries([key]);
  });
}
