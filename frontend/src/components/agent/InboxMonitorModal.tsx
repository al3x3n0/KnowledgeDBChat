/**
 * The research-inbox monitor form.
 *
 * Module scope, like its siblings — declared in the page's render body it was
 * remounted on every page render, resetting the form below.
 */

import { XCircle } from 'lucide-react';
import React, { useState } from 'react';

import Button from '../common/Button';

interface InboxMonitorModalProps {
  onClose: () => void;
  createInboxMonitorMutation: any;
}

export const InboxMonitorModal: React.FC<InboxMonitorModalProps> = ({
  onClose,
  createInboxMonitorMutation,
}) => {

  const [name, setName] = useState('Research Inbox Monitor');
  const [customer, setCustomer] = useState('');
  const [customerContext, setCustomerContext] = useState('');
  const [intervalMinutes, setIntervalMinutes] = useState(60);
  const [maxDocuments, setMaxDocuments] = useState(8);
  const [maxPapers, setMaxPapers] = useState(8);
  const [includeDocuments, setIncludeDocuments] = useState(true);
  const [includeArxiv, setIncludeArxiv] = useState(true);
  const [monitorQueriesText, setMonitorQueriesText] = useState('');
  const [persistArtifacts, setPersistArtifacts] = useState(false);
  const [autoAddToReadingList, setAutoAddToReadingList] = useState(false);
  const [readingListName, setReadingListName] = useState('Research Inbox');
  const [followUpAutonomyMode, setFollowUpAutonomyMode] = useState<'manual_only' | 'auto_launch_safe' | 'queue_for_approval'>('manual_only');
  const [runImmediately, setRunImmediately] = useState(true);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const preferSources: string[] = [];
    if (includeDocuments) preferSources.push('documents');
    if (includeArxiv) preferSources.push('arxiv');

    const monitorQueries = monitorQueriesText
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean);

    const goal =
      (customer || '').trim().length > 0
        ? `Continuously monitor for customer-relevant updates (${customer.trim()}) and file them into the Research Inbox.`
        : 'Continuously monitor for customer-relevant updates and file them into the Research Inbox.';

    createInboxMonitorMutation.mutate({
      name,
      job_type: 'monitor',
      goal,
      config: {
        deterministic_runner: 'research_inbox_monitor',
        automation_profile: 'balanced',
        automation_policy: {
          follow_up_review_mode: followUpAutonomyMode,
          allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
        },
        customer: customer.trim(),
        customer_context: customerContext,
        prefer_sources: preferSources,
        monitor_queries: monitorQueries,
        max_documents: maxDocuments,
        max_papers: maxPapers,
        interval_minutes: intervalMinutes,
        persist_artifacts: persistArtifacts,
        auto_add_to_reading_list: autoAddToReadingList,
        reading_list_name: readingListName,
        follow_up_autonomy: {
          mode: followUpAutonomyMode,
          allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
        },
      },
      schedule_type: 'continuous',
      start_immediately: runImmediately,
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-xl max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold">Create Research Inbox Monitor</h2>
              <p className="text-sm text-gray-500">Runs continuously and files new items into your inbox</p>
            </div>
            <Button variant="ghost" size="sm" onClick={() => onClose()}>
              <XCircle className="w-5 h-5" />
            </Button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Interval (minutes)</label>
                <input
                  type="number"
                  min={1}
                  max={1440}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={intervalMinutes}
                  onChange={(e) => setIntervalMinutes(parseInt(e.target.value || '60', 10))}
                />
              </div>
              <div className="flex items-center gap-2 pt-6">
                <input
                  type="checkbox"
                  checked={runImmediately}
                  onChange={(e) => setRunImmediately(e.target.checked)}
                />
                <span className="text-sm text-gray-700">Run immediately</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Customer (optional)</label>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={customer}
                onChange={(e) => setCustomer(e.target.value)}
                placeholder="Acme Corp"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Customer context (optional)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={3}
                value={customerContext}
                onChange={(e) => setCustomerContext(e.target.value)}
                placeholder="What we care about, constraints, success metrics..."
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={includeDocuments}
                  onChange={(e) => setIncludeDocuments(e.target.checked)}
                />
                <span className="text-sm text-gray-700">Search internal documents</span>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={includeArxiv}
                  onChange={(e) => setIncludeArxiv(e.target.checked)}
                />
                <span className="text-sm text-gray-700">Search arXiv</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Accepted follow-up policy</label>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={followUpAutonomyMode}
                onChange={(e) => setFollowUpAutonomyMode(e.target.value as any)}
              >
                <option value="manual_only">Manual only</option>
                <option value="auto_launch_safe">Auto-launch safe deep dives</option>
                <option value="queue_for_approval">Queue for approval before launch</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                Safe auto-launch only applies to bounded built-in deep-dive follow-ups. Repo patch chains stay manual.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Max KB docs / run</label>
                <input
                  type="number"
                  min={0}
                  max={50}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={maxDocuments}
                  onChange={(e) => setMaxDocuments(parseInt(e.target.value || '8', 10))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Max papers / run</label>
                <input
                  type="number"
                  min={0}
                  max={50}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={maxPapers}
                  onChange={(e) => setMaxPapers(parseInt(e.target.value || '8', 10))}
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Monitor queries (optional, one per line)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={3}
                value={monitorQueriesText}
                onChange={(e) => setMonitorQueriesText(e.target.value)}
                placeholder="e.g.\nLLM safety evaluation\ncustomer SLA latency"
              />
              <p className="text-xs text-gray-500 mt-1">If empty, queries are derived from goal + customer profile/context.</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={persistArtifacts}
                  onChange={(e) => setPersistArtifacts(e.target.checked)}
                />
                <span className="text-sm text-gray-700">Persist weekly brief doc</span>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={autoAddToReadingList}
                  onChange={(e) => setAutoAddToReadingList(e.target.checked)}
                />
                <span className="text-sm text-gray-700">Auto-add docs to reading list</span>
              </div>
            </div>

            {autoAddToReadingList ? (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Reading list name</label>
                <input
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={readingListName}
                  onChange={(e) => setReadingListName(e.target.value)}
                />
              </div>
            ) : null}

            <div className="flex justify-end gap-3 pt-4 border-t">
              <Button type="button" variant="secondary" onClick={() => onClose()}>
                Cancel
              </Button>
              <Button type="submit" disabled={createInboxMonitorMutation.isLoading}>
                {createInboxMonitorMutation.isLoading ? 'Creating…' : 'Create Monitor'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default InboxMonitorModal;
