/**
 * A research campaign the assistant drafted, offered for review.
 *
 * Some chat messages are not questions. "Run a campaign to find out whether X"
 * asks for work to be started, and a wall of prose is the wrong answer to it.
 * When the backend recognises that shape it attaches a draft to the reply and
 * this renders it.
 *
 * **It is an offer, not an action.** A campaign spends a budget of agent jobs
 * autonomously; inferring that from a sentence and launching it is the kind of
 * helpfulness nobody asks for twice. Everything here is editable and nothing
 * runs until Launch is pressed.
 *
 * Dismissing is local to this view rather than persisted: the draft stays on
 * the message, so scrolling back still shows what was offered.
 */

import { AlertCircle, Loader2, Rocket, X } from 'lucide-react';
import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

import { apiClient } from '../../services/api';
import type { ResearchCampaignDraft } from '../../types';

export interface CampaignDraftWidgetProps {
  draft: ResearchCampaignDraft;
}

const CampaignDraftWidget: React.FC<CampaignDraftWidgetProps> = ({ draft }) => {
  const navigate = useNavigate();
  const [dismissed, setDismissed] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [launchedId, setLaunchedId] = useState<string | null>(null);
  const [name, setName] = useState(draft.name);
  const [goal, setGoal] = useState(draft.goal);
  const [budget, setBudget] = useState(draft.max_jobs);
  // Seeds are droppable but not editable in place: rewriting a question reads
  // better once the campaign exists than inside a chat bubble.
  const [items, setItems] = useState(draft.items || []);

  if (dismissed) return null;

  const launch = async () => {
    if (!goal.trim()) return;
    setLaunching(true);
    try {
      const campaign = await apiClient.createResearchCampaign({
        name: name.trim() || goal.trim().slice(0, 60),
        goal: goal.trim(),
        items,
        max_jobs: budget,
      });
      setLaunchedId(campaign.id);
      toast.success(
        `Campaign started: "${campaign.name}". The scheduler picks it up within five minutes and runs one job at a time.`,
      );
    } catch {
      // apiClient surfaces the error itself; the widget stays open so an
      // edited draft is not lost to a failed request.
    } finally {
      setLaunching(false);
    }
  };

  if (launchedId) {
    return (
      <div className="mt-3 rounded-lg border border-primary-500/40 bg-primary-500/5 p-3">
        <div className="flex items-center gap-2 text-sm text-gray-900">
          <Rocket className="w-4 h-4 text-primary-600" />
          <span className="font-medium">{name}</span>
          <span className="text-gray-600">is running</span>
        </div>
        <p className="mt-1 text-xs text-gray-600">
          {items.length} seed question{items.length === 1 ? '' : 's'}, up to {budget}{' '}
          job{budget === 1 ? '' : 's'}. It advances on its own from here.
        </p>
        <button
          type="button"
          onClick={() => navigate('/autonomous-agents')}
          className="mt-2 text-xs text-primary-600 hover:text-primary-700 underline"
        >
          Watch its jobs
        </button>
      </div>
    );
  }

  return (
    <div
      className="mt-3 rounded-lg border border-gray-300 bg-gray-100 p-3"
      data-testid="campaign-draft-widget"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 text-sm text-gray-900">
          <Rocket className="w-4 h-4 text-primary-600" />
          <span className="font-medium">Start a research campaign?</span>
        </div>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          className="p-1 rounded hover:bg-gray-200 text-gray-500 hover:text-gray-800"
          title="Dismiss"
          aria-label="Dismiss campaign draft"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <p className="mt-1 text-xs text-gray-600">
        Drafted from what you asked. Nothing runs until you launch it.
      </p>

      <label className="block mt-3 text-xs text-gray-600">
        Name
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="mt-1 w-full bg-white border border-gray-300 rounded px-2 py-1 text-sm"
        />
      </label>

      <label className="block mt-2 text-xs text-gray-600">
        Goal — what the campaign has to settle
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          rows={2}
          className="mt-1 w-full bg-white border border-gray-300 rounded px-2 py-1 text-sm resize-y"
        />
      </label>
      {!goal.trim() && (
        <p className="mt-1 flex items-center gap-1 text-xs text-amber-700">
          <AlertCircle className="w-3 h-3" />
          A campaign without a goal runs until it exhausts its budget.
        </p>
      )}

      {items.length > 0 && (
        <div className="mt-3">
          <div className="text-xs text-gray-600 mb-1">
            Starts with {items.length} question{items.length === 1 ? '' : 's'}
          </div>
          <ul className="space-y-1">
            {items.map((item, idx) => (
              <li
                key={`${idx}-${item.title.slice(0, 20)}`}
                className="flex items-start justify-between gap-2 text-xs text-gray-700 bg-white border border-gray-200 rounded px-2 py-1"
              >
                <span className="min-w-0 break-words">{item.title}</span>
                <button
                  type="button"
                  onClick={() => setItems(items.filter((_, i) => i !== idx))}
                  className="text-gray-500 hover:text-gray-800 flex-shrink-0"
                  aria-label={`Remove ${item.title}`}
                >
                  <X className="w-3 h-3" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-3 flex items-center gap-3">
        <label className="text-xs text-gray-600">
          Job budget
          <input
            type="number"
            min={1}
            max={500}
            value={budget}
            onChange={(e) => setBudget(Math.max(1, Number(e.target.value) || 1))}
            className="ml-2 w-16 bg-white border border-gray-300 rounded px-2 py-1 text-sm"
          />
        </label>
        <button
          type="button"
          onClick={launch}
          disabled={launching || !goal.trim()}
          className="ml-auto inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-primary-600 text-gray-50 text-sm hover:bg-primary-700 disabled:opacity-50"
        >
          {launching ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Rocket className="w-3.5 h-3.5" />
          )}
          Launch
        </button>
      </div>
    </div>
  );
};

export default CampaignDraftWidget;
