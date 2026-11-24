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

import { Cpu, Loader2, Plus, Rocket, X } from 'lucide-react';
import React, { useRef, useState } from 'react';
import toast from 'react-hot-toast';

import { apiClient } from '../../services/api';
import type { ResearchCampaign } from '../../types';
import { CPU_CO_DESIGN_CAMPAIGN } from '../../utils/cpuCoDesignStudy';
import Button from '../common/Button';

export interface NewCampaignModalProps {
  onClose: () => void;
  onCreated?: (campaign: ResearchCampaign) => void;
}

interface StartingQuestion {
  id: number;
  title: string;
  detail: string;
}

const DEFAULT_CAMPAIGN_BUDGET = 8;

const NewCampaignModal: React.FC<NewCampaignModalProps> = ({ onClose, onCreated }) => {
  const [name, setName] = useState('');
  const [goal, setGoal] = useState('');
  const [questions, setQuestions] = useState<StartingQuestion[]>([{ id: 0, title: '', detail: '' }]);
  const nextQuestionId = useRef(1);
  const [budgetInput, setBudgetInput] = useState(String(DEFAULT_CAMPAIGN_BUDGET));
  // Off by default: a campaign that approves its own plans runs work no person
  // has looked at. The server refuses limits over the deployment's ceiling.
  const [selfApprove, setSelfApprove] = useState(false);
  const [maxStages, setMaxStages] = useState(2);
  const [maxMinutes, setMaxMinutes] = useState(60);
  const [saving, setSaving] = useState(false);

  const budget = Number(budgetInput);
  const validBudget = /^\d+$/.test(budgetInput) && Number.isInteger(budget) && budget >= 1 && budget <= 500;
  const startingQuestions = questions.filter((question) => question.title.trim());
  const initialJobs = Math.max(1, startingQuestions.length);
  const canSubmit = Boolean(goal.trim()) && validBudget && !saving &&
    questions.every((question) => question.title.trim() || !question.detail.trim());

  const updateQuestion = (id: number, fields: Partial<Pick<StartingQuestion, 'title' | 'detail'>>) => {
    setQuestions((current) => current.map((question) => question.id === id ? { ...question, ...fields } : question));
  };

  const useCpuCoDesignTemplate = () => {
    const hasDraft = name.trim() || goal.trim() || budget !== DEFAULT_CAMPAIGN_BUDGET ||
      questions.some((question) => question.title.trim() || question.detail.trim());
    if (hasDraft && !window.confirm('Replace the current name, goal, starting questions, and job limit with the CPU co-design template?')) return;
    const template = CPU_CO_DESIGN_CAMPAIGN;
    setName(template.name);
    setGoal(template.goal);
    setQuestions(template.items.map((question) => ({ ...question, id: nextQuestionId.current++ })));
    setBudgetInput(String(template.max_jobs));
  };

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
        items: startingQuestions.map(({ title, detail }) => ({
          title: title.trim(),
          ...(detail.trim() && { detail: detail.trim() }),
        })),
        max_jobs: budget,
        plan_approval: selfApprove
          ? { max_stages: maxStages, max_minutes: maxMinutes }
          : null,
      });
      toast.success('Campaign started');
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
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4">
      <form
        role="dialog"
        aria-modal="true"
        aria-labelledby="new-campaign-title"
        onSubmit={(event) => {
          event.preventDefault();
          if (canSubmit) void submit();
        }}
        className="flex max-h-[90vh] w-full max-w-lg flex-col overflow-hidden rounded-xl border border-gray-300 bg-white shadow-xl"
      >
        <div className="flex shrink-0 items-center justify-between border-b border-gray-200 px-5 py-4">
          <div className="flex items-center gap-2">
            <Rocket className="h-5 w-5 text-primary-600" />
            <h2 id="new-campaign-title" className="text-lg font-semibold text-gray-900">New Campaign</h2>
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

        <div className="min-h-0 space-y-4 overflow-y-auto px-5 py-4">
          <p className="text-xs text-gray-600">
            A campaign pursues one goal across a budget of agent jobs, raising its
            own follow-up questions from what it finds.
          </p>

          <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-gray-300 bg-gray-100 p-3">
            <div>
              <p className="text-sm font-medium text-gray-900">CPU co-design template</p>
              <p className="text-xs text-gray-600">Compare kernel and load queue changes in gem5.</p>
            </div>
            <Button
              type="button"
              size="sm"
              variant="secondary"
              aria-label="Use CPU co-design template"
              onClick={useCpuCoDesignTemplate}
              icon={<Cpu className="h-4 w-4" />}
            >
              Use template
            </Button>
          </div>

          <label className="block text-sm">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">
              Research goal
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
              State a question the campaign can answer with evidence.
            </span>
          </label>

          <label className="block text-sm">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">
              Name <span className="normal-case text-gray-400">(optional)</span>
            </span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
                maxLength={300}
              placeholder="Taken from the goal if left empty"
              className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
            />
          </label>

          <fieldset className="text-sm">
            <legend className="text-xs font-medium uppercase tracking-wide text-gray-500">
              Starting questions <span className="normal-case text-gray-400">(optional)</span>
            </legend>
            <div className="mt-1 space-y-2">
              {questions.map((question, idx) => (
                <div key={question.id} className="rounded-lg border border-gray-300 bg-gray-100 p-2">
                  <div className="flex items-start gap-2">
                    <textarea
                      value={question.title}
                      onChange={(e) => updateQuestion(question.id, { title: e.target.value })}
                      aria-label={`Starting question ${idx + 1}`}
                      maxLength={300}
                      rows={question.title.length > 55 ? 2 : 1}
                      placeholder="One question an agent job can answer"
                      className="w-full resize-y rounded-md border border-gray-300 bg-gray-50 px-2 py-1.5 text-sm"
                    />
                    {questions.length > 1 && (
                      <button
                        type="button"
                        onClick={() => setQuestions((current) => current.filter((item) => item.id !== question.id))}
                        className="shrink-0 rounded p-1.5 text-gray-600 hover:bg-gray-200 hover:text-gray-900"
                        aria-label={`Remove question ${idx + 1}`}
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                  <details className="mt-1.5">
                    <summary className="cursor-pointer text-xs text-primary-700">
                      {question.detail.trim() ? 'Edit instructions' : 'Add instructions'}
                    </summary>
                    <textarea
                      aria-label={`Instructions for question ${idx + 1}`}
                      value={question.detail}
                      onChange={(e) => updateQuestion(question.id, { detail: e.target.value })}
                      rows={4}
                      placeholder="Include benchmark controls, measurements, and required evidence"
                      className="mt-1.5 w-full resize-y rounded-md border border-gray-300 bg-gray-50 px-2 py-1.5 text-sm"
                    />
                  </details>
                  {question.detail.trim() && !question.title.trim() && (
                    <p className="mt-1 text-xs text-amber-300">Enter a starting question to include these instructions.</p>
                  )}
                </div>
              ))}
            </div>
            <button
              type="button"
              onClick={() => {
                const id = nextQuestionId.current++;
                setQuestions((current) => [...current, { id, title: '', detail: '' }]);
              }}
              className="mt-2 inline-flex items-center gap-1 text-xs text-primary-600 hover:text-primary-700"
            >
              <Plus className="h-3 w-3" />
              Add question
            </button>
            <span className="mt-1 block text-xs text-gray-500">
              The campaign starts from your goal if you leave this empty. It can add follow-up questions from each job’s findings.
            </span>
          </fieldset>

          <label className="block text-sm">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">
              Maximum agent jobs
            </span>
            <input
              type="number"
              min={1}
              max={500}
              step={1}
              required
              value={budgetInput}
              onChange={(e) => setBudgetInput(e.target.value)}
              aria-invalid={!validBudget}
              aria-describedby="campaign-budget-help"
              className="mt-1 w-24 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
            />
            <span id="campaign-budget-help" className="mt-1 block text-xs text-gray-500">
              {validBudget ? 'Pipeline stages and follow-up questions also use this budget.' : 'Enter a whole number from 1 to 500.'}
            </span>
          </label>

          <div className="text-sm">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={selfApprove}
                onChange={(e) => setSelfApprove(e.target.checked)}
              />
              <span>Approve its own plans</span>
            </label>
            <span className="mt-1 block text-xs text-gray-500">
              When a job proposes a follow-up plan, the campaign may start it without
              asking you, if it is this small. Anything larger, and any checkpoint the
              plan declares, still waits for you.
            </span>
            {selfApprove && (
              <div className="mt-2 flex items-end gap-4">
                <label className="block">
                  <span className="text-xs text-gray-500">Up to stages</span>
                  <input
                    type="number"
                    aria-label="Up to stages"
                    min={1}
                    max={4}
                    value={maxStages}
                    onChange={(e) => setMaxStages(Math.max(1, Number(e.target.value) || 1))}
                    className="mt-1 block w-20 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                  />
                </label>
                <label className="block">
                  <span className="text-xs text-gray-500">Up to minutes</span>
                  <input
                    type="number"
                    aria-label="Up to minutes"
                    min={1}
                    max={120}
                    value={maxMinutes}
                    onChange={(e) => setMaxMinutes(Math.max(1, Number(e.target.value) || 1))}
                    className="mt-1 block w-24 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                  />
                </label>
              </div>
            )}
          </div>
          <section aria-label="Before starting" className="space-y-1 rounded-lg border border-gray-300 bg-gray-100 p-3 text-sm">
            <h3 className="font-medium text-gray-900">Before starting</h3>
            <p className="text-xs text-gray-600">
              {startingQuestions.length > 0
                ? `${startingQuestions.length} starting ${startingQuestions.length === 1 ? 'question' : 'questions'} · at least ${initialJobs} ${initialJobs === 1 ? 'job' : 'jobs'} to investigate all of them`
                : 'The research goal is the first question and uses one job to start.'}
            </p>
            <p className="text-xs text-gray-600">
              {selfApprove
                ? `Automatic approval limit · ${maxStages} ${maxStages === 1 ? 'stage' : 'stages'} and ${maxMinutes} min per plan`
                : 'You review each proposed plan.'}
            </p>
          </section>
        </div>

        <div className="shrink-0 space-y-2 border-t border-gray-200 px-5 py-4">
          {validBudget && initialJobs > budget && (
            <p role="status" className="text-xs text-amber-700 dark:text-amber-300">
              A {budget}-job budget can leave {initialJobs - budget} or more starting questions uninvestigated.
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={!canSubmit}>
              {saving ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Rocket className="mr-2 h-4 w-4" />
              )}
              Start campaign
            </Button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default NewCampaignModal;
