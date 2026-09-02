/**
 * Start a customer-research run, from a template or a chain.
 *
 * Module scope, like its siblings — declared in the page's render body it was
 * remounted on every page render, resetting the form below.
 */

import { Brain, XCircle } from 'lucide-react';
import React, { useMemo, useState } from 'react';
import toast from 'react-hot-toast';

import Button from '../common/Button';

interface CustomerResearchModalProps {
  onClose: () => void;
  createFromTemplateMutation: any;
  createFromChainMutation: any;
  templatesData: any;
  chainsData: any;
}

export const CustomerResearchModal: React.FC<CustomerResearchModalProps> = ({
  onClose,
  createFromTemplateMutation,
  createFromChainMutation,
  templatesData,
  chainsData,
}) => {

  const [name, setName] = useState(`Customer Research — ${new Date().toLocaleDateString()}`);
  const [goal, setGoal] = useState('');
  const [customerContext, setCustomerContext] = useState('');
  const [persistArtifacts, setPersistArtifacts] = useState(false);
  const [addToReadingList, setAddToReadingList] = useState(true);
  const [readingListName, setReadingListName] = useState('Customer Research');
  const [runDeepDive, setRunDeepDive] = useState(false);
  const [sourcePreference, setSourcePreference] = useState<'documents_first' | 'balanced' | 'arxiv_first'>('documents_first');
  const [maxDocuments, setMaxDocuments] = useState(12);
  const [maxPapers, setMaxPapers] = useState(8);

  const templates = (templatesData as any)?.templates || [];
  const deepDiveTemplate = templates.find((t: any) => t?.name === 'customer_research_scout_deep_dive');
  const scoutTemplate = templates.find((t: any) => t?.name === 'customer_research_scout');
  const template =
    (runDeepDive ? deepDiveTemplate : null) ||
    scoutTemplate ||
    deepDiveTemplate ||
    templates.find((t: any) => t?.category === 'research');

  const deepDiveChain = useMemo(() => {
    const chains = (chainsData as any)?.chains || [];
    return chains.find((c: any) => c?.name === 'customer_research_scout_deep_dive_chain') || null;
  }, [chainsData]);

  const preferSources =
    sourcePreference === 'documents_first'
      ? ['documents', 'arxiv']
      : sourcePreference === 'arxiv_first'
        ? ['arxiv', 'documents']
        : ['documents', 'arxiv'];

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    if (!template?.id) {
      toast.error('Customer research template not available');
      return;
    }
    if (!goal.trim()) {
      toast.error('Goal is required');
      return;
    }

    const chainConfig = runDeepDive && !deepDiveTemplate
      ? {
          trigger_condition: 'on_complete' as const,
          inherit_results: true,
          inherit_config: true,
          child_jobs: [
            {
              name: 'Customer Research — Deep Dive',
              job_type: 'research' as const,
              goal:
                'Deep-dive using inherited results from the scout job. Focus on the top internal documents and any high-signal papers. ' +
                'Output: (1) 3-5 hypotheses, (2) risks/unknowns, (3) minimal experiment plan (metrics + timeline), ' +
                'and (4) a short brief document.',
              config: {
                prefer_sources: ['documents'],
                max_documents: 6,
                max_papers: 0,
              },
              max_iterations: 6,
              max_tool_calls: 40,
              max_llm_calls: 12,
              max_runtime_minutes: 10,
            },
          ],
        }
      : undefined;

    // Prefer starting from the built-in chain definition if available: it creates a first job and chains the deep dive.
    if (runDeepDive && deepDiveChain?.id) {
      createFromChainMutation.mutate({
        chain_definition_id: String(deepDiveChain.id),
        name_prefix: name.trim(),
        variables: { goal: goal.trim() },
        config_overrides: {
          customer_context: customerContext.trim() || undefined,
          persist_artifacts: !!persistArtifacts,
          reading_list_name: addToReadingList ? (readingListName.trim() || 'Customer Research') : undefined,
          prefer_sources: preferSources,
          max_documents: Math.max(1, Math.min(200, Number(maxDocuments) || 12)),
          max_papers: Math.max(1, Math.min(200, Number(maxPapers) || 8)),
        },
        start_immediately: true,
      } as any);
      return;
    }

    createFromTemplateMutation.mutate({
      template_id: template.id,
      name: name.trim(),
      goal: goal.trim(),
      start_immediately: true,
      chain_config: chainConfig as any,
      config: {
        customer_context: customerContext.trim() || undefined,
        persist_artifacts: !!persistArtifacts,
        reading_list_name: addToReadingList ? (readingListName.trim() || 'Customer Research') : undefined,
        prefer_sources: preferSources,
        max_documents: Math.max(1, Math.min(200, Number(maxDocuments) || 12)),
        max_papers: Math.max(1, Math.min(200, Number(maxPapers) || 8)),
      },
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl">
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary-100 text-primary-600">
              <Brain className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Customer Research</h2>
              <p className="text-sm text-gray-500">
                Uses the deployment customer profile + optional context to run a tailored research loop.
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={() => onClose()}>
            <XCircle className="w-5 h-5" />
          </Button>
        </div>

        <form onSubmit={handleCreate} className="p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Job Name</label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Source preference</label>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={sourcePreference}
                onChange={(e) => setSourcePreference(e.target.value as any)}
              >
                <option value="documents_first">Prefer internal documents first</option>
                <option value="balanced">Balanced</option>
                <option value="arxiv_first">Prefer arXiv first</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Goal</label>
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              rows={3}
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="E.g., 'Summarize the state of the art on X and propose 3 experiments tailored to our constraints.'"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Customer context (optional)</label>
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              rows={3}
              value={customerContext}
              onChange={(e) => setCustomerContext(e.target.value)}
              placeholder="Any extra details not captured in the deployment customer profile."
            />
            <div className="mt-1 text-xs text-gray-500">
              This is combined with the deployment customer profile (Admin → AI Hub) when generating the plan.
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max internal docs</label>
              <input
                type="number"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={maxDocuments}
                onChange={(e) => setMaxDocuments(parseInt(e.target.value || '0', 10))}
                min={1}
                max={200}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max papers</label>
              <input
                type="number"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={maxPapers}
                onChange={(e) => setMaxPapers(parseInt(e.target.value || '0', 10))}
                min={1}
                max={200}
              />
            </div>
          </div>

          <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-2">
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={addToReadingList}
                onChange={(e) => setAddToReadingList(e.target.checked)}
              />
              Allow adding relevant documents to a reading list
            </label>
            {addToReadingList && (
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Reading list name</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={readingListName}
                  onChange={(e) => setReadingListName(e.target.value)}
                />
              </div>
            )}
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={persistArtifacts}
                onChange={(e) => setPersistArtifacts(e.target.checked)}
              />
              Allow saving a brief document (optional)
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={runDeepDive}
                onChange={(e) => setRunDeepDive(e.target.checked)}
              />
              Auto-run a deep-dive follow-up job
            </label>
            <div className="text-xs text-gray-500">
              If enabled, the agent may create a short “Customer Research Brief” document in the knowledge base.
            </div>
          </div>

          <div className="flex justify-end gap-3 pt-4 border-t">
            <Button type="button" variant="secondary" onClick={() => onClose()}>
              Cancel
            </Button>
            <Button type="submit" disabled={createFromTemplateMutation.isLoading}>
              {createFromTemplateMutation.isLoading ? 'Starting...' : 'Start Research Job'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default CustomerResearchModal;
