/**
 * The "new job" form.
 *
 * It was declared inside AutonomousAgentsPage's render body, which meant a new
 * component type on every page render: React unmounted and remounted it rather
 * than reconciling, and the form state below went back to its defaults. The
 * page polls every ten seconds, so a half-filled form rarely survived to be
 * submitted. Declared once, at module scope, it keeps what you typed.
 */

import React, { useState } from 'react';
import toast from 'react-hot-toast';

import type { AgentJobCreate, AgentJobType } from '../../types';
import Button from '../common/Button';

interface CreateJobModalProps {
  onClose: () => void;
  /** The page's create mutation: its loading state drives the submit button. */
  createMutation: any;
}

export const CreateJobModal: React.FC<CreateJobModalProps> = ({ onClose, createMutation }) => {
  const [formData, setFormData] = useState<Partial<AgentJobCreate>>({
    name: '',
    job_type: 'research',
    goal: '',
    max_iterations: 50,
    max_tool_calls: 200,
    max_llm_calls: 100,
    max_runtime_minutes: 30,
    start_immediately: true,
    config: {},
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.name || !formData.goal) {
      toast.error('Name and goal are required');
      return;
    }
    createMutation.mutate(formData as AgentJobCreate);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-4">Create Autonomous Job</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="My Research Job"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Job Type</label>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={formData.job_type}
                onChange={(e) => setFormData({ ...formData, job_type: e.target.value as AgentJobType })}
              >
                <option value="research">Research</option>
                <option value="analysis">Analysis</option>
                <option value="data_analysis">Data Analysis (ETL, Charts, Diagrams)</option>
                <option value="monitor">Monitor</option>
                <option value="synthesis">Synthesis</option>
                <option value="knowledge_expansion">Knowledge Expansion</option>
                <option value="custom">Custom</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Goal</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={4}
                value={formData.goal}
                onChange={(e) => setFormData({ ...formData, goal: e.target.value })}
                placeholder="Research the latest developments in transformer architectures..."
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Max Iterations</label>
                <input
                  type="number"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={formData.max_iterations}
                  onChange={(e) => setFormData({ ...formData, max_iterations: parseInt(e.target.value) })}
                  min={1}
                  max={1000}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Max Runtime (min)</label>
                <input
                  type="number"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={formData.max_runtime_minutes}
                  onChange={(e) => setFormData({ ...formData, max_runtime_minutes: parseInt(e.target.value) })}
                  min={1}
                  max={480}
                />
              </div>
            </div>


            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-700 mb-2">LLM Routing (optional)</div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Tier</label>
                  <select
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={String((formData.config as any)?.llm_tier || '')}
                    onChange={(e) => {
                      const tier = e.target.value;
                      const cfg = { ...((formData.config as any) || {}) };
                      if (!tier) {
                        delete (cfg as any).llm_tier;
                      } else {
                        (cfg as any).llm_tier = tier;
                      }
                      setFormData({ ...formData, config: cfg });
                    }}
                  >
                    <option value="">(default)</option>
                    <option value="fast">fast</option>
                    <option value="balanced">balanced</option>
                    <option value="deep">deep</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Fallback tiers (comma)</label>
                  <input
                    type="text"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={String(((formData.config as any)?.llm_fallback_tiers || []).join(', '))}
                    onChange={(e) => {
                      const raw = e.target.value;
                      const arr = raw
                        .split(',')
                        .map((s) => s.trim())
                        .filter(Boolean);
                      const cfg = { ...((formData.config as any) || {}) };
                      if (arr.length === 0) {
                        delete (cfg as any).llm_fallback_tiers;
                      } else {
                        (cfg as any).llm_fallback_tiers = arr;
                      }
                      setFormData({ ...formData, config: cfg });
                    }}
                    placeholder="balanced, fast"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Timeout (sec)</label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={String((formData.config as any)?.llm_timeout_seconds ?? '')}
                    onChange={(e) => {
                      const v = e.target.value;
                      const cfg = { ...((formData.config as any) || {}) };
                      if (!v) {
                        delete (cfg as any).llm_timeout_seconds;
                      } else {
                        (cfg as any).llm_timeout_seconds = parseInt(v, 10);
                      }
                      setFormData({ ...formData, config: cfg });
                    }}
                    min={2}
                    max={600}
                    placeholder="120"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Max tokens cap</label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={String((formData.config as any)?.llm_max_tokens_cap ?? '')}
                    onChange={(e) => {
                      const v = e.target.value;
                      const cfg = { ...((formData.config as any) || {}) };
                      if (!v) {
                        delete (cfg as any).llm_max_tokens_cap;
                      } else {
                        (cfg as any).llm_max_tokens_cap = parseInt(v, 10);
                      }
                      setFormData({ ...formData, config: cfg });
                    }}
                    min={64}
                    max={20000}
                    placeholder="2000"
                  />
                </div>
              </div>
              <div className="mt-2 text-xs text-gray-500">
                Uses feature flags <span className="font-mono">llm_provider_* / llm_model_*</span> for tier resolution; falls back on failures.
              </div>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="start_immediately"
                checked={formData.start_immediately}
                onChange={(e) => setFormData({ ...formData, start_immediately: e.target.checked })}
              />
              <label htmlFor="start_immediately" className="text-sm text-gray-700">
                Start immediately
              </label>
            </div>

            <div className="flex justify-end gap-3 pt-4 border-t">
              <Button type="button" variant="secondary" onClick={onClose}>
                Cancel
              </Button>
              <Button type="submit" disabled={createMutation.isLoading}>
                {createMutation.isLoading ? 'Creating...' : 'Create Job'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default CreateJobModal;
