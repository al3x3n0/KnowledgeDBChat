/**
 * Create a job from a saved template.
 *
 * At module scope for the same reason as its siblings: declared inside the
 * page's render body it was a new component type on every render, so React
 * remounted it and the form state below reset — including on the page's
 * ten-second poll.
 */

import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

import type { AgentJobTemplate } from '../../types';
import Button from '../common/Button';

interface CreateFromTemplateModalProps {
  template: AgentJobTemplate;
  onClose: () => void;
  createFromTemplateMutation: any;
  /** The github/gitlab document sources a coding template can be pointed at. */
  codeSources: any[];
}

export const CreateFromTemplateModal: React.FC<CreateFromTemplateModalProps> = ({
  template,
  onClose,
  createFromTemplateMutation,
  codeSources,
}) => {
  const navigate = useNavigate();

  const [name, setName] = useState(`${template.display_name} - ${new Date().toLocaleDateString()}`);
  const [goal, setGoal] = useState(template.default_goal || '');
  const [configText, setConfigText] = useState(
    template.default_config ? JSON.stringify(template.default_config, null, 2) : ''
  );
  const templateRunner = String((template.default_config as any)?.deterministic_runner || '').toLowerCase();
  const isClaudeBackendTemplate = template.name === 'claude_code_backend';
  const isCodePatchTemplate =
    template.name === 'code_patch_proposer' ||
    isClaudeBackendTemplate ||
    (String(template.category || '').toLowerCase() === 'code' && templateRunner === 'code_patch_proposer');
  const [selectedTargetSourceId, setSelectedTargetSourceId] = useState<string>('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name) {
      toast.error('Name is required');
      return;
    }
    let parsedConfig: any | undefined = undefined;
    if (configText.trim()) {
      try {
        parsedConfig = JSON.parse(configText);
      } catch (err) {
        toast.error('Config must be valid JSON');
        return;
      }
    }
    createFromTemplateMutation.mutate({
      template_id: template.id,
      name,
      goal: goal !== template.default_goal ? goal : undefined,
      config: parsedConfig,
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-1">Create from Template</h2>
          <p className="text-sm text-gray-500 mb-4">{template.display_name}</p>
          <form onSubmit={handleSubmit} className="space-y-4">
            {codeSources.length === 0 && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
                <div className="text-sm font-medium text-amber-800">No Git code sources found</div>
                <p className="mt-1 text-xs text-amber-700">
                  Ingest a GitHub/GitLab repository first, then return to Quick Start.
                </p>
                <div className="mt-2">
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    onClick={() => navigate('/documents')}
                  >
                    Open Documents
                  </Button>
                </div>
              </div>
            )}
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
              <label className="block text-sm font-medium text-gray-700 mb-1">Goal</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={4}
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
              />
            </div>

            {isCodePatchTemplate && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Target code source</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={selectedTargetSourceId}
                  onChange={(e) => {
                    const id = e.target.value;
                    setSelectedTargetSourceId(id);
                    try {
                      const obj = configText.trim() ? JSON.parse(configText) : {};
                      const next = { ...(obj || {}), source_id: id } as any;
                      if (isClaudeBackendTemplate) {
                        if (!next.search_query) next.search_query = 'backend';
                        if (!Array.isArray(next.file_paths)) next.file_paths = [];
                        if (!Array.isArray(next.commands)) next.commands = [];
                        if (next.auto_commands_from_project_profile === undefined) {
                          next.auto_commands_from_project_profile = true;
                        }
                        if (next.auto_commands_profile_max_files === undefined) {
                          next.auto_commands_profile_max_files = 300;
                        }
                      }
                      setConfigText(JSON.stringify(next, null, 2));
                    } catch {
                      const fallback: any = { source_id: id };
                      if (isClaudeBackendTemplate) {
                        fallback.search_query = 'backend';
                        fallback.file_paths = [];
                        fallback.commands = [];
                        fallback.auto_commands_from_project_profile = true;
                        fallback.auto_commands_profile_max_files = 300;
                      }
                      setConfigText(JSON.stringify(fallback, null, 2));
                    }
                  }}
                >
                  <option value="">Select a git source…</option>
                  {codeSources.map((s: any) => (
                    <option key={String(s.id)} value={String(s.id)}>
                      {String(s.name || s.id)}
                    </option>
                  ))}
                </select>
                <div className="mt-1 text-xs text-gray-500">
                  This should be a GitHub/GitLab document source (code ingested into the KB).
                </div>
              </div>
            )}

            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-2">Template Configuration</p>
              <div className="flex gap-4 text-sm text-gray-600">
                <span>Max {template.default_max_iterations} iterations</span>
                <span>{template.default_max_runtime_minutes} min runtime</span>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-700 mb-2">LLM Routing (applies to config JSON)</div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Tier</label>
                  <select
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={(() => {
                      try {
                        const obj = configText.trim() ? JSON.parse(configText) : {};
                        return String((obj as any)?.llm_tier || '');
                      } catch {
                        return '';
                      }
                    })()}
                    onChange={(e) => {
                      const tier = e.target.value;
                      try {
                        const obj = configText.trim() ? JSON.parse(configText) : {};
                        const next = { ...(obj || {}) } as any;
                        if (!tier) delete next.llm_tier;
                        else next.llm_tier = tier;
                        setConfigText(JSON.stringify(next, null, 2));
                      } catch {
                        const next: any = {};
                        if (tier) next.llm_tier = tier;
                        setConfigText(JSON.stringify(next, null, 2));
                      }
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
                    value={(() => {
                      try {
                        const obj = configText.trim() ? JSON.parse(configText) : {};
                        return String(((obj as any)?.llm_fallback_tiers || []).join(', '));
                      } catch {
                        return '';
                      }
                    })()}
                    onChange={(e) => {
                      const raw = e.target.value;
                      const arr = raw
                        .split(',')
                        .map((s) => s.trim())
                        .filter(Boolean);
                      try {
                        const obj = configText.trim() ? JSON.parse(configText) : {};
                        const next = { ...(obj || {}) } as any;
                        if (arr.length === 0) delete next.llm_fallback_tiers;
                        else next.llm_fallback_tiers = arr;
                        setConfigText(JSON.stringify(next, null, 2));
                      } catch {
                        const next: any = {};
                        if (arr.length) next.llm_fallback_tiers = arr;
                        setConfigText(JSON.stringify(next, null, 2));
                      }
                    }}
                    placeholder="balanced, fast"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Timeout (sec)</label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={(() => {
                      try {
                        const obj = configText.trim() ? JSON.parse(configText) : {};
                        const v = (obj as any)?.llm_timeout_seconds;
                        return v === undefined || v === null ? '' : String(v);
                      } catch {
                        return '';
                      }
                    })()}
                    onChange={(e) => {
                      const v = e.target.value;
                      try {
                        const obj = configText.trim() ? JSON.parse(configText) : {};
                        const next = { ...(obj || {}) } as any;
                        if (!v) delete next.llm_timeout_seconds;
                        else next.llm_timeout_seconds = parseInt(v, 10);
                        setConfigText(JSON.stringify(next, null, 2));
                      } catch {
                        const next: any = {};
                        if (v) next.llm_timeout_seconds = parseInt(v, 10);
                        setConfigText(JSON.stringify(next, null, 2));
                      }
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
                    value={(() => {
                      try {
                        const obj = configText.trim() ? JSON.parse(configText) : {};
                        const v = (obj as any)?.llm_max_tokens_cap;
                        return v === undefined || v === null ? '' : String(v);
                      } catch {
                        return '';
                      }
                    })()}
                    onChange={(e) => {
                      const v = e.target.value;
                      try {
                        const obj = configText.trim() ? JSON.parse(configText) : {};
                        const next = { ...(obj || {}) } as any;
                        if (!v) delete next.llm_max_tokens_cap;
                        else next.llm_max_tokens_cap = parseInt(v, 10);
                        setConfigText(JSON.stringify(next, null, 2));
                      } catch {
                        const next: any = {};
                        if (v) next.llm_max_tokens_cap = parseInt(v, 10);
                        setConfigText(JSON.stringify(next, null, 2));
                      }
                    }}
                    min={64}
                    max={20000}
                    placeholder="2000"
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Config (JSON)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={6}
                value={configText}
                onChange={(e) => setConfigText(e.target.value)}
                placeholder='{"key":"value"}'
              />
              {template.name === 'code_patch_proposer' && (
                <div className="mt-1 text-xs text-gray-500">
                  Required: <span className="font-mono">source_id</span> (UUID of a git document source).
                </div>
              )}
            </div>

            <div className="flex justify-end gap-3 pt-4 border-t">
              <Button type="button" variant="secondary" onClick={() => onClose()}>
                Cancel
              </Button>
              <Button type="submit" disabled={createFromTemplateMutation.isLoading}>
                {createFromTemplateMutation.isLoading ? 'Creating...' : 'Create Job'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default CreateFromTemplateModal;
