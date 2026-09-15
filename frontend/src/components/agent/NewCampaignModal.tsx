/**
 * Start a research campaign.
 *
 * Replaces the "Customer Research" button, which opened a modal bound to two
 * job templates -- `customer_research_scout` and `..._deep_dive` -- that do
 * not exist in the database. Its submit path began with
 * `if (!template?.id) { toast.error('Customer research template not
 * available'); return; }`, so every attempt ended there: a button that opened
 * a form that could not be submitted.
 *
 * A campaign is the thing that entry point was reaching for. It takes a goal
 * and pursues it across a budget of agent jobs, spawning follow-up questions
 * from what it finds, rather than running one templated job.
 *
 * The same shape as the campaign widget the chat offers, so the two entry
 * points do not disagree about what a campaign needs: a goal it is judged
 * against, seeds to start from, and a budget.
 */

import { Loader2, Plus, Rocket, X } from 'lucide-react';
import React, { useState } from 'react';
import toast from 'react-hot-toast';

import { apiClient } from '../../services/api';
import type { ResearchCampaign } from '../../types';
import Button from '../common/Button';

export interface NewCampaignModalProps {
  onClose: () => void;
  onCreated?: (campaign: ResearchCampaign) => void;
}

const NewCampaignModal: React.FC<NewCampaignModalProps> = ({ onClose, onCreated }) => {
  const [name, setName] = useState('');
  const [goal, setGoal] = useState('');
  const [seeds, setSeeds] = useState<string[]>(['']);
  const [budget, setBudget] = useState(8);
  const [saving, setSaving] = useState(false);

  const trimmedSeeds = seeds.map((s) => s.trim()).filter(Boolean);
  const canSubmit = Boolean(goal.trim()) && !saving;

  const submit = async () => {
    if (!goal.trim()) {
      toast.error('A campaign needs a goal: it is what completion is judged against');
      return;
    }
    setSaving(true);
    try {
      const campaign = await apiClient.createResearchCampaign({
        name: name.trim() || goal.trim().slice(0, 60),
        goal: goal.trim(),
        items: trimmedSeeds.map((title) => ({ title })),
        max_jobs: budget,
      });
      toast.success(
        `Campaign started: "${campaign.name}". The scheduler picks it up within five minutes and runs one job at a time.`,
      );
      onCreated?.(campaign);
      onClose();
    } catch {
      // apiClient surfaces the error; keep the form open so the text typed
      // into it is not lost to a failed request.
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl bg-white border border-gray-300 shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-200 px-5 py-4">
          <div className="flex items-center gap-2">
            <Rocket className="h-5 w-5 text-primary-600" />
            <h2 className="text-lg font-semibold text-gray-900">New Campaign</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-gray-500 hover:bg-gray-200 hover:text-gray-900"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-4 px-5 py-4">
          <p className="text-xs text-gray-600">
            A campaign pursues one goal across a budget of agent jobs, raising its
            own follow-up questions from what it finds.
          </p>

          <label className="block text-sm">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">
              Goal
            </span>
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              rows={3}
              autoFocus
              placeholder="Establish whether a stride prefetcher speeds up the dotprod kernel"
              className="mt-1 w-full resize-y rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
            />
            <span className="mt-1 block text-xs text-gray-500">
              What the campaign has to settle — a claim, not a topic. Completion is
              judged against this, so a campaign without one runs until its budget
              is gone.
            </span>
          </label>

          <label className="block text-sm">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">
              Name <span className="normal-case text-gray-400">(optional)</span>
            </span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Taken from the goal if left empty"
              className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
            />
          </label>

          <div className="text-sm">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">
              Starting questions <span className="normal-case text-gray-400">(optional)</span>
            </span>
            <div className="mt-1 space-y-2">
              {seeds.map((seed, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <input
                    value={seed}
                    onChange={(e) =>
                      setSeeds(seeds.map((s, i) => (i === idx ? e.target.value : s)))
                    }
                    placeholder="One question an agent job can answer"
                    className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                  />
                  {seeds.length > 1 && (
                    <button
                      type="button"
                      onClick={() => setSeeds(seeds.filter((_, i) => i !== idx))}
                      className="flex-shrink-0 rounded p-1 text-gray-500 hover:bg-gray-200 hover:text-gray-900"
                      aria-label="Remove question"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              ))}
            </div>
            <button
              type="button"
              onClick={() => setSeeds([...seeds, ''])}
              className="mt-2 inline-flex items-center gap-1 text-xs text-primary-600 hover:text-primary-700"
            >
              <Plus className="h-3 w-3" />
              Add question
            </button>
            <span className="mt-1 block text-xs text-gray-500">
              Left empty, the campaign takes the goal as its first question.
              Either way it raises follow-ups from what each job finds.
            </span>
          </div>

          <label className="block text-sm">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">
              Job budget
            </span>
            <input
              type="number"
              min={1}
              max={500}
              value={budget}
              onChange={(e) => setBudget(Math.min(500, Math.max(1, Number(e.target.value) || 1)))}
              className="mt-1 w-24 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
            />
            <span className="mt-1 block text-xs text-gray-500">
              The campaign stops here even if the goal is not settled.
            </span>
          </label>
        </div>

        <div className="flex justify-end gap-2 border-t border-gray-200 px-5 py-4">
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={!canSubmit}>
            {saving ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Rocket className="mr-2 h-4 w-4" />
            )}
            Start campaign
          </Button>
        </div>
      </div>
    </div>
  );
};

export default NewCampaignModal;
