/**
 * Start a job from a saved chain.
 *
 * Declared at module scope, not inside the page's render body: a component
 * redefined every render is a new type, and React remounts rather than
 * reconciles it — which reset the form below on every ten-second poll.
 */

import { GitBranch, XCircle } from 'lucide-react';
import React, { useMemo, useState } from 'react';
import toast from 'react-hot-toast';

import type { AgentJobChainDefinition } from '../../types';
import Button from '../common/Button';

interface StartChainModalProps {
  chain: AgentJobChainDefinition;
  onClose: () => void;
  createFromChainMutation: any;
}

export const StartChainModal: React.FC<StartChainModalProps> = ({
  chain,
  onClose,
  createFromChainMutation,
}) => {

  const defaultPrefix = `${chain.display_name} — ${new Date().toLocaleDateString()}`;
  const [namePrefix, setNamePrefix] = useState(defaultPrefix);
  const [startImmediately, setStartImmediately] = useState(true);
  const [configOverridesRaw, setConfigOverridesRaw] = useState<string>('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const variableKeys = useMemo(() => {
    const keys = new Set<string>();
    const steps = (chain as any)?.chain_steps || [];
    const re = /\{([a-zA-Z0-9_]+)\}/g;
    for (const s of steps) {
      const tmpl = String(s?.goal_template || '');
      let m: RegExpExecArray | null;
      while ((m = re.exec(tmpl)) !== null) {
        if (m[1]) keys.add(m[1]);
      }
    }
    return Array.from(keys).sort();
  }, [chain]);

  const [variables, setVariables] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    for (const k of variableKeys) initial[k] = '';
    return initial;
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!namePrefix.trim()) {
      toast.error('Name prefix is required');
      return;
    }
    const payloadVars: Record<string, string> = {};
    for (const k of variableKeys) {
      const v = String(variables[k] || '').trim();
      if (v) payloadVars[k] = v;
    }

    let config_overrides: Record<string, any> | undefined = undefined;
    const raw = (configOverridesRaw || '').trim();
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          config_overrides = parsed;
        } else {
          toast.error('Config overrides must be a JSON object');
          return;
        }
      } catch {
        toast.error('Invalid JSON in config overrides');
        return;
      }
    }
    createFromChainMutation.mutate({
      chain_definition_id: chain.id,
      name_prefix: namePrefix.trim(),
      variables: payloadVars,
      config_overrides,
      start_immediately: startImmediately,
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg">
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
              <GitBranch className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Start Chain</h2>
              <p className="text-sm text-gray-500">{chain.display_name}</p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={() => onClose()}>
            <XCircle className="w-5 h-5" />
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name prefix</label>
            <input
              type="text"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={namePrefix}
              onChange={(e) => setNamePrefix(e.target.value)}
            />
            <div className="mt-1 text-xs text-gray-500">
              Used to name each step job (e.g., “{namePrefix} - Step 1”).
            </div>
          </div>

          {variableKeys.length > 0 ? (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-800 mb-2">Variables</div>
              <div className="space-y-2">
                {variableKeys.map((k: string) => (
                  <div key={k}>
                    <label className="block text-xs font-medium text-gray-600 mb-1">{k}</label>
                    <input
                      type="text"
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={variables[k] || ''}
                      onChange={(e) => setVariables((prev) => ({ ...prev, [k]: e.target.value }))}
                      placeholder={`Value for {${k}}`}
                    />
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-sm text-gray-600 bg-gray-50 border border-gray-200 rounded-lg p-3">
              This chain has no variables.
            </div>
          )}

          <label className="flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={startImmediately}
              onChange={(e) => setStartImmediately(e.target.checked)}
            />
            Start immediately
          </label>

          <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
            <button
              type="button"
              className="text-sm font-medium text-gray-800"
              onClick={() => setShowAdvanced((v) => !v)}
            >
              Advanced: config overrides (JSON)
            </button>
            {showAdvanced && (
              <div className="mt-2 space-y-2">
                <div className="text-xs text-gray-500">
                  Optional. Passed as <span className="font-mono">config_overrides</span> to the chain start request.
                </div>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs font-mono"
                  rows={6}
                  value={configOverridesRaw}
                  onChange={(e) => setConfigOverridesRaw(e.target.value)}
                  placeholder='{"latex_project_id":"...","source_id":"...","search_query":"..."}'
                />
              </div>
            )}
          </div>

          <div className="flex justify-end gap-3 pt-4 border-t">
            <Button type="button" variant="secondary" onClick={() => onClose()}>
              Cancel
            </Button>
            <Button type="submit" disabled={createFromChainMutation.isLoading}>
              {createFromChainMutation.isLoading ? 'Starting…' : 'Start Chain'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default StartChainModal;
