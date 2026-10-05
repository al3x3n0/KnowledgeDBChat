import React from 'react';
import { ThumbsDown } from 'lucide-react';
import { useQuery } from 'react-query';

import Button from '../common/Button';
import { apiClient } from '../../services/api';
import { ResearchInboxRejectionReason } from '../../types';

/**
 * Rejecting an item, with the reason it is being rejected for.
 *
 * "Not my subject" and "my subject, done badly" are different sentences, and
 * the monitor profile learns different things from them: only the first is
 * evidence about the words. Before this, every rejection taught the words, so
 * turning down a weak paper on your own topic taught the profile to hide that
 * topic. The choices and their effects come from the backend
 * (`/research/inbox/rejection-reasons`) rather than being restated here, so
 * what a person is told a click will do is what the learner actually does.
 *
 * If that request fails — an older backend, an offline moment — this falls back
 * to a plain Reject that sends no reason, which the learner reads exactly as it
 * read every rejection before reasons existed. Triage must not become
 * impossible because an explanation is unavailable.
 */
export function RejectWithReason({
  disabled,
  onReject,
  className = '',
  label = 'Reject',
}: {
  disabled?: boolean;
  onReject: (reason?: string) => void;
  className?: string;
  label?: string;
}) {
  const [open, setOpen] = React.useState(false);

  const { data, isLoading } = useQuery(
    'research-inbox-rejection-reasons',
    () => apiClient.getResearchInboxRejectionReasons(),
    { staleTime: 60 * 60 * 1000, retry: false }
  );

  const reasons: ResearchInboxRejectionReason[] = Array.isArray(data?.reasons)
    ? data!.reasons
    : [];

  // "Still loading" is not "unavailable". Collapsing the two meant a click
  // landing before the vocabulary arrived rejected with no reason at all, so
  // whoever clicked fastest silently lost the choice.
  if (reasons.length === 0 && !isLoading) {
    return (
      <Button
        size="sm"
        variant="secondary"
        disabled={disabled}
        className={className}
        onClick={() => onReject(undefined)}
      >
        <ThumbsDown className="w-4 h-4 mr-1" />
        {label}
      </Button>
    );
  }

  if (!open) {
    return (
      <Button
        size="sm"
        variant="secondary"
        disabled={disabled}
        className={className}
        onClick={() => setOpen(true)}
      >
        <ThumbsDown className="w-4 h-4 mr-1" />
        {label}
      </Button>
    );
  }

  return (
    <div className={`border border-gray-300 rounded-lg p-2 grid gap-1 ${className}`}>
      <p className="text-xs text-gray-500">Reject because…</p>
      {isLoading ? <p className="text-xs text-gray-500">Loading choices…</p> : null}
      {reasons.map((reason) => (
        <button
          key={reason.key}
          type="button"
          disabled={disabled}
          className="text-left px-2 py-1 rounded hover:bg-gray-200 disabled:opacity-50"
          onClick={() => {
            setOpen(false);
            onReject(reason.key);
          }}
        >
          <span className="text-sm">{reason.label}</span>
          {/* What the click will teach. A choice whose effect is unstated is a
              choice made blind — and three of these deliberately teach nothing. */}
          <span className="block text-xs text-gray-500">{reason.effect}</span>
        </button>
      ))}
      <button
        type="button"
        className="text-xs text-gray-500 justify-self-start px-2 py-1"
        onClick={() => setOpen(false)}
      >
        Cancel
      </button>
    </div>
  );
}

export default RejectWithReason;
