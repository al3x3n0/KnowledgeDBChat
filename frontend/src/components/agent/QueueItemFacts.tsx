/**
 * One rendering of the facts a review queue item carries.
 *
 * Two surfaces show the same queue: the Runs page's operator queue (a flat,
 * fleet-wide list from `/agent-jobs/checkpoint-queue`) and the Control Plane's
 * review list (grouped by control run, from `/agent-control-plane/reviews`).
 * Those are genuinely different *aggregations* — one is flat and fleet-wide,
 * the other is per-run and also carries opportunity-sourced reviews — so they
 * are not one endpoint and should not be forced into one.
 *
 * What they do share is the item. Both draw on `_build_checkpoint_queue_items`
 * server-side, and both then rendered its facts themselves, which is how they
 * drifted: the same overdue item appeared as a red badge on one screen and as
 * grey text reading "SLA: overdue" on the other. An operator triaging across
 * both saw one queue described two ways.
 *
 * So: one component, two adapters. The field names differ slightly between the
 * two response shapes (`item_type` vs `review_type`, `queue_key` vs
 * `queue_item_key`), and the adapters are the only place that knows.
 */

import React from 'react';

export interface QueueItemFacts {
  itemType?: string | null;
  status?: string | null;
  reasonLabel?: string | null;
  slaBucket?: string | null;
  escalationLevel?: string | null;
  ageMinutes?: number | null;
  priorityScore?: number | null;
  createdAt?: string | null;
}

/** From `/agent-jobs/checkpoint-queue` (the Runs operator queue). */
export const factsFromCheckpointQueueItem = (item: any): QueueItemFacts => ({
  itemType: item?.item_type,
  status: item?.status,
  reasonLabel: item?.reason_label,
  slaBucket: item?.sla_bucket,
  escalationLevel: item?.escalation_level,
  ageMinutes: item?.age_minutes,
  priorityScore: item?.priority_score,
  createdAt: item?.created_at,
});

/** From `/agent-control-plane/reviews` (the Control Plane's run reviews). */
export const factsFromControlReview = (review: any): QueueItemFacts => ({
  itemType: review?.item_type ?? review?.review_type,
  status: review?.status,
  reasonLabel: review?.reason_label,
  slaBucket: review?.sla_bucket,
  escalationLevel: review?.escalation_level,
  ageMinutes: review?.age_minutes,
  priorityScore: review?.priority_score,
  createdAt: review?.created_at,
});

const humanize = (value: string) => value.replace(/_/g, ' ');

const ITEM_TYPE_CLASS: Record<string, string> = {
  policy_review: 'bg-rose-100 text-rose-800',
  budget_review: 'bg-amber-100 text-amber-800',
};

const SLA_CLASS: Record<string, string> = {
  overdue: 'bg-rose-100 text-rose-800',
  at_risk: 'bg-amber-100 text-amber-800',
};

const ESCALATION_CLASS: Record<string, string> = {
  high: 'bg-rose-50 text-rose-700 border border-rose-200',
  medium: 'bg-amber-50 text-amber-700 border border-amber-200',
};

const Badge: React.FC<{ className: string; children: React.ReactNode }> = ({ className, children }) => (
  <span className={`text-xs px-2 py-1 rounded ${className}`}>{children}</span>
);

export const QueueItemFactsRow: React.FC<{
  facts: QueueItemFacts;
  /** The item type is a heading on one surface and a badge on the other. */
  showItemType?: boolean;
  className?: string;
}> = ({ facts, showItemType = true, className }) => {
  const age = facts.ageMinutes;
  const urgency = facts.priorityScore;
  return (
    <div className={`flex items-center flex-wrap gap-2 ${className || ''}`}>
      {showItemType && facts.itemType ? (
        <Badge className={ITEM_TYPE_CLASS[facts.itemType] || 'bg-blue-100 text-blue-800'}>
          {humanize(facts.itemType)}
        </Badge>
      ) : null}
      {facts.status ? <Badge className="bg-gray-100 text-gray-700">{facts.status}</Badge> : null}
      {facts.reasonLabel ? (
        <Badge className="bg-gray-200 text-gray-700">{facts.reasonLabel}</Badge>
      ) : null}
      {facts.slaBucket ? (
        <Badge className={SLA_CLASS[facts.slaBucket] || 'bg-emerald-100 text-emerald-800'}>
          {humanize(facts.slaBucket)}
        </Badge>
      ) : null}
      {facts.escalationLevel ? (
        <Badge className={ESCALATION_CLASS[facts.escalationLevel] || 'bg-gray-100 text-gray-600 border border-gray-200'}>
          {facts.escalationLevel}
        </Badge>
      ) : null}
      {age !== undefined && age !== null ? (
        <span className="text-xs text-gray-500">Age: {age}m</span>
      ) : null}
      {urgency !== undefined && urgency !== null ? (
        <span className="text-xs text-gray-500">Urgency: {urgency}</span>
      ) : null}
      {facts.createdAt ? (
        <span className="text-xs text-gray-500">{new Date(facts.createdAt).toLocaleString()}</span>
      ) : null}
    </div>
  );
};

export default QueueItemFactsRow;
