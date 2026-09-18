import React from 'react';
import { render, screen } from '@testing-library/react';

import QueueItemFactsRow, {
  factsFromCheckpointQueueItem,
  factsFromControlReview,
} from '../QueueItemFacts';

/**
 * The same queue item reaches two surfaces through two different endpoints, and
 * before this component each rendered its facts itself. They drifted: an
 * overdue item showed as a red badge on the Runs queue and as grey text reading
 * "SLA: overdue" on the Control Plane, so one queue was described two ways to
 * the person triaging it.
 *
 * These tests pin the thing that matters -- that both shapes produce the same
 * facts -- rather than the markup, which is free to change as long as it
 * changes for both.
 */
describe('queue item facts', () => {
  const checkpointQueueItem = {
    item_type: 'approval_checkpoint',
    status: 'pending',
    reason_label: 'Tool needs approval',
    sla_bucket: 'overdue',
    escalation_level: 'high',
    age_minutes: 42,
    priority_score: 7,
    created_at: '2026-03-17T11:00:00Z',
  };

  /** The control plane renames two fields and keeps the rest. */
  const controlReview = {
    review_type: 'approval_checkpoint',
    item_type: 'approval_checkpoint',
    status: 'pending',
    reason_label: 'Tool needs approval',
    sla_bucket: 'overdue',
    escalation_level: 'high',
    age_minutes: 42,
    priority_score: 7,
    created_at: '2026-03-17T11:00:00Z',
  };

  it('reads the same facts from both response shapes', () => {
    expect(factsFromControlReview(controlReview))
      .toEqual(factsFromCheckpointQueueItem(checkpointQueueItem));
  });

  it('falls back to review_type when the control shape omits item_type', () => {
    const { item_type, ...withoutItemType } = controlReview;
    expect(factsFromControlReview(withoutItemType).itemType).toBe('approval_checkpoint');
  });

  it('renders an overdue item the same way whichever surface it came from', () => {
    const { container: fromQueue } = render(
      <QueueItemFactsRow facts={factsFromCheckpointQueueItem(checkpointQueueItem)} />
    );
    const { container: fromControl } = render(
      <QueueItemFactsRow facts={factsFromControlReview(controlReview)} />
    );
    expect(fromControl.innerHTML).toBe(fromQueue.innerHTML);
  });

  it('gives overdue and at-risk distinct treatment from on-track', () => {
    const cls = (bucket: string) => {
      const { container } = render(<QueueItemFactsRow facts={{ slaBucket: bucket }} />);
      return (container.querySelector('span') as HTMLElement).className;
    };
    const overdue = cls('overdue');
    const atRisk = cls('at_risk');
    const onTrack = cls('on_track');
    expect(new Set([overdue, atRisk, onTrack]).size).toBe(3);
  });

  it('omits facts the item does not carry rather than rendering blanks', () => {
    render(<QueueItemFactsRow facts={{ itemType: 'job_recovery' }} />);
    expect(screen.queryByText(/Age:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Urgency:/)).not.toBeInTheDocument();
  });
});
