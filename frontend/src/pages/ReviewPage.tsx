/**
 * Everything waiting on you, in one place.
 *
 * The decisions a person owes were spread over five pages under three doors,
 * and tool approvals were on the Admin page only. This lists them together.
 * It owns none of them: each section links to the page where that kind of
 * decision is made. Tool approvals are the exception, since for anyone who
 * is not an admin there was no such page.
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { AlertTriangle, ArrowRight, Check, Loader2, RefreshCw, X } from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';
import { ReviewItem, ReviewSection } from '../types';
import { waitingFor } from '../utils/reviewInbox';

const REVIEW_QUERY = ['reviewInbox'];

const ReviewPage: React.FC = () => {
  const queryClient = useQueryClient();
  const { data, isLoading, isFetching, isError, refetch } = useQuery(
    REVIEW_QUERY,
    () => apiClient.getReviewInbox(),
    { staleTime: 15000, refetchOnWindowFocus: true }
  );

  const refresh = () => queryClient.invalidateQueries(REVIEW_QUERY);

  // Approving is consent for the action to happen, so once nobody else's
  // approval is outstanding it runs. With an admin's half still owed, it waits.
  const approve = useMutation(
    async (auditId: string) => {
      const approved = await apiClient.approveToolAudit(auditId);
      if (approved.approval_status !== 'approved') return { ran: false };
      await apiClient.runToolAudit(auditId);
      return { ran: true };
    },
    {
      onSuccess: (result) => {
        toast.success(result.ran ? 'Approved and run' : 'Approved. Waiting for an admin.');
        refresh();
      },
      onError: (e: any) => {
        toast.error(e?.response?.data?.detail || 'Could not approve');
        refresh();
      },
    }
  );

  const reject = useMutation((auditId: string) => apiClient.rejectToolAudit(auditId), {
    onSuccess: () => {
      toast.success('Rejected');
      refresh();
    },
    onError: (e: any) => {
      toast.error(e?.response?.data?.detail || 'Could not reject');
    },
  });

  const sections: ReviewSection[] = data?.sections || [];
  const waiting = sections.filter((s) => s.count > 0 || s.error);
  const clear = sections.filter((s) => s.count === 0 && !s.error);
  const busy = approve.isLoading || reject.isLoading;

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Review</h1>
          <p className="text-gray-600">
            {isLoading
              ? 'Checking what is waiting on you…'
              : data && data.total > 0
              ? `${data.total} ${data.total === 1 ? 'thing is' : 'things are'} waiting on you`
              : 'Nothing is waiting on you'}
          </p>
        </div>
        <button
          type="button"
          onClick={() => refetch()}
          className="inline-flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-100"
          aria-label="Refresh"
        >
          {isFetching ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
          Refresh
        </button>
      </div>

      {isError && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700">
          <AlertTriangle className="w-4 h-4" />
          The review list could not be loaded.
        </div>
      )}

      <div className="space-y-4">
        {waiting.map((section) => (
          <section key={section.key} className="rounded-lg border border-gray-200 bg-white" aria-label={section.title}>
            <header className="flex items-start justify-between gap-4 p-4 border-b border-gray-200">
              <div>
                <h2 className="text-base font-semibold text-gray-900">
                  {section.title}
                  {!section.error && (
                    <span className="ml-2 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">
                      {section.count}
                    </span>
                  )}
                </h2>
                <p className="text-sm text-gray-600">{section.description}</p>
              </div>
              {section.link !== '/review' && (
                <Link
                  to={section.link}
                  className="shrink-0 inline-flex items-center gap-1 text-sm text-primary-600 hover:underline"
                >
                  Open <ArrowRight className="w-4 h-4" />
                </Link>
              )}
            </header>

            {section.error ? (
              <p className="p-4 text-sm text-red-700">
                {section.error} Whether anything is waiting here is unknown.
              </p>
            ) : (
              <ul className="divide-y divide-gray-200">
                {section.items.map((item) => (
                  <ReviewRow
                    key={item.id}
                    item={item}
                    busy={busy}
                    onApprove={section.key === 'tool_approvals' ? () => approve.mutate(item.id) : undefined}
                    onReject={section.key === 'tool_approvals' ? () => reject.mutate(item.id) : undefined}
                  />
                ))}
                {section.count > section.items.length && (
                  <li className="px-4 py-2 text-xs text-gray-500">
                    and {section.count - section.items.length} more
                  </li>
                )}
              </ul>
            )}
          </section>
        ))}
      </div>

      {clear.length > 0 && !isLoading && (
        <p className="mt-6 text-sm text-gray-500">
          Nothing waiting in: {clear.map((s) => s.title).join(', ')}.
        </p>
      )}
    </div>
  );
};

const ReviewRow: React.FC<{
  item: ReviewItem;
  busy: boolean;
  onApprove?: () => void;
  onReject?: () => void;
}> = ({ item, busy, onApprove, onReject }) => {
  const waited = waitingFor(item.waiting_since);
  const title = item.link ? (
    <Link to={item.link} className="font-medium text-gray-900 hover:underline">
      {item.title}
    </Link>
  ) : (
    <span className="font-medium text-gray-900">{item.title}</span>
  );

  return (
    <li className="flex items-center justify-between gap-4 px-4 py-3">
      <div className="min-w-0">
        <div className="truncate text-sm">{title}</div>
        {item.detail && <div className="truncate text-xs text-gray-600">{item.detail}</div>}
      </div>
      <div className="flex shrink-0 items-center gap-3">
        {waited && <span className="text-xs text-gray-500">{waited}</span>}
        {onApprove && onReject && (
          <>
            <button
              type="button"
              disabled={busy}
              onClick={onApprove}
              className="inline-flex items-center gap-1 rounded-lg bg-primary-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-primary-700 disabled:opacity-50"
            >
              <Check className="w-3 h-3" /> Approve
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={onReject}
              className="inline-flex items-center gap-1 rounded-lg border border-gray-300 px-2.5 py-1 text-xs font-medium text-gray-700 hover:bg-gray-100 disabled:opacity-50"
            >
              <X className="w-3 h-3" /> Reject
            </button>
          </>
        )}
      </div>
    </li>
  );
};

export default ReviewPage;
