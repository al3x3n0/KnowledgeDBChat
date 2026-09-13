/**
 * Quick start: a role-based workflow run.
 *
 * Module scope, like its siblings — declared in the page's render body it was
 * remounted on every page render, resetting the form below.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';

import Button from '../common/Button';

/** Where this form remembers its last answers, between openings. */
const ROLE_WORKFLOW_QS_PREFS_KEY = 'autonomous_agents_role_workflow_qs_prefs_v1';

interface QuickStartRoleWorkflowModalProps {
  onClose: () => void;
  quickStartRoleWorkflowMutation: any;
  templateRecommendGoal: string;
}

export const QuickStartRoleWorkflowModal: React.FC<QuickStartRoleWorkflowModalProps> = ({
  onClose,
  quickStartRoleWorkflowMutation,
  templateRecommendGoal,
}) => {

  const [name, setName] = useState(`Role Workflow - ${new Date().toLocaleDateString()}`);
  const [goal, setGoal] = useState(
    templateRecommendGoal.trim() || 'Investigate contradictory evidence and deliver a validated action plan'
  );
  const [rolesText, setRolesText] = useState('researcher_documents\nresearcher_arxiv\nanalyst\nsynthesizer');
  const [maxAgents, setMaxAgents] = useState<number>(4);
  const [memoryProfile, setMemoryProfile] = useState<'off' | 'minimal' | 'balanced' | 'evidence' | 'synthesis'>('balanced');
  const [approvalMode, setApprovalMode] = useState<'high_impact' | 'none'>('high_impact');
  const [executionMode, setExecutionMode] = useState<'plan_and_execute' | 'adaptive'>('plan_and_execute');
  const [extractMemoryOnFailure, setExtractMemoryOnFailure] = useState<boolean>(true);
  const [memoryFailedTypesText, setMemoryFailedTypesText] = useState<string>('pattern\nlesson\ninsight');
  const [memoryCompletedTypesText, setMemoryCompletedTypesText] = useState<string>('');
  const [configOverridesText, setConfigOverridesText] = useState<string>('{}');

  const parseRoles = useCallback((raw: string): string[] => {
    const out: string[] = [];
    const seen = new Set<string>();
    const rows = String(raw || '')
      .split('\n')
      .map((line) => line.trim().toLowerCase().replace(/[-\s]+/g, '_'))
      .filter(Boolean);
    for (const role of rows) {
      if (!/^[a-z0-9_:-]{2,120}$/.test(role)) continue;
      if (seen.has(role)) continue;
      seen.add(role);
      out.push(role);
      if (out.length >= 12) break;
    }
    return out;
  }, []);

  const rolePreview = useMemo(() => parseRoles(rolesText), [rolesText, parseRoles]);
  const parseMemoryTypes = useCallback((raw: string): string[] => {
    const out: string[] = [];
    const seen = new Set<string>();
    const allowed = new Set(['finding', 'insight', 'pattern', 'lesson', 'fact', 'preference', 'context', 'summary']);
    const rows = String(raw || '')
      .replace(/\n/g, ',')
      .split(',')
      .map((line) => line.trim().toLowerCase().replace(/[-\s]+/g, '_'))
      .filter(Boolean);
    for (const row of rows) {
      if (!allowed.has(row)) continue;
      if (seen.has(row)) continue;
      seen.add(row);
      out.push(row);
      if (out.length >= 12) break;
    }
    return out;
  }, []);
  const memoryFailedTypesPreview = useMemo(
    () => parseMemoryTypes(memoryFailedTypesText),
    [memoryFailedTypesText, parseMemoryTypes]
  );
  const memoryCompletedTypesPreview = useMemo(
    () => parseMemoryTypes(memoryCompletedTypesText),
    [memoryCompletedTypesText, parseMemoryTypes]
  );
  const effectiveMaxAgents = Math.max(1, Math.min(maxAgents || 1, 12));

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(ROLE_WORKFLOW_QS_PREFS_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw || '{}') as Record<string, any>;
      const roles = String(parsed?.roles_text || '').trim();
      const mem = String(parsed?.memory_profile || '').trim().toLowerCase();
      const appr = String(parsed?.approval_mode || '').trim().toLowerCase();
      const mode = String(parsed?.execution_mode || '').trim().toLowerCase().replace(/[-\s]+/g, '_');
      const extractOnFailure = parsed?.extract_memory_on_failure;
      const failedTypesText = String(parsed?.memory_failed_types_text || '').trim();
      const completedTypesText = String(parsed?.memory_completed_types_text || '').trim();
      const mx = Number(parsed?.max_agents || 0);
      const overrides = String(parsed?.config_overrides_text || '');
      if (roles) setRolesText(roles);
      if (mem === 'off' || mem === 'minimal' || mem === 'balanced' || mem === 'evidence' || mem === 'synthesis') {
        setMemoryProfile(mem);
      }
      if (appr === 'high_impact' || appr === 'none') setApprovalMode(appr);
      if (mode === 'plan_and_execute' || mode === 'adaptive') setExecutionMode(mode);
      if (typeof extractOnFailure === 'boolean') setExtractMemoryOnFailure(extractOnFailure);
      if (Object.prototype.hasOwnProperty.call(parsed, 'memory_failed_types_text')) {
        setMemoryFailedTypesText(String(parsed?.memory_failed_types_text || ''));
      } else if (failedTypesText) {
        setMemoryFailedTypesText(failedTypesText);
      }
      if (Object.prototype.hasOwnProperty.call(parsed, 'memory_completed_types_text')) {
        setMemoryCompletedTypesText(String(parsed?.memory_completed_types_text || ''));
      } else if (completedTypesText) {
        setMemoryCompletedTypesText(completedTypesText);
      }
      if (Number.isFinite(mx) && mx > 0) setMaxAgents(Math.max(1, Math.min(mx, 12)));
      if (overrides) setConfigOverridesText(overrides);
    } catch {
      // Ignore malformed local preferences.
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        ROLE_WORKFLOW_QS_PREFS_KEY,
        JSON.stringify({
          roles_text: rolesText || '',
          memory_profile: memoryProfile,
          approval_mode: approvalMode,
          execution_mode: executionMode,
          extract_memory_on_failure: extractMemoryOnFailure,
          memory_failed_types_text: memoryFailedTypesText || '',
          memory_completed_types_text: memoryCompletedTypesText || '',
          max_agents: effectiveMaxAgents,
          config_overrides_text: configOverridesText || '{}',
        })
      );
    } catch {
      // Ignore localStorage write failures.
    }
  }, [
    rolesText,
    memoryProfile,
    approvalMode,
    executionMode,
    extractMemoryOnFailure,
    memoryFailedTypesText,
    memoryCompletedTypesText,
    effectiveMaxAgents,
    configOverridesText,
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim()) {
      toast.error('Goal is required');
      return;
    }
    if (!rolePreview.length) {
      toast.error('At least one valid role is required');
      return;
    }

    let overrides: Record<string, any> | undefined = undefined;
    const rawOverrides = configOverridesText.trim();
    if (rawOverrides && rawOverrides !== '{}') {
      try {
        const parsed = JSON.parse(rawOverrides);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          overrides = parsed as Record<string, any>;
        } else {
          toast.error('Config overrides must be a JSON object');
          return;
        }
      } catch {
        toast.error('Invalid JSON in config overrides');
        return;
      }
    }

    quickStartRoleWorkflowMutation.mutate({
      name: name.trim() || undefined,
      goal: goal.trim(),
      roles: rolePreview,
      max_agents: effectiveMaxAgents,
      memory_profile: memoryProfile,
      approval_mode: approvalMode,
      execution_mode: executionMode,
      extract_memory_on_failure: extractMemoryOnFailure,
      memory_failed_types: memoryFailedTypesPreview.length ? memoryFailedTypesPreview : undefined,
      memory_completed_types: memoryCompletedTypesPreview.length ? memoryCompletedTypesPreview : undefined,
      start_immediately: true,
      config_overrides: overrides,
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-1">Quick Start Role Workflow</h2>
          <p className="text-sm text-gray-500 mb-4">
            Launch a multi-agent role workflow with fan-in synthesis, approval gates, and memory profile presets.
          </p>
          <form onSubmit={handleSubmit} className="space-y-4">
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
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Max agents</label>
                <input
                  type="number"
                  min={1}
                  max={12}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={effectiveMaxAgents}
                  onChange={(e) => setMaxAgents(parseInt(e.target.value || '4', 10))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Memory profile</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={memoryProfile}
                  onChange={(e) => setMemoryProfile(e.target.value as any)}
                >
                  <option value="off">Off</option>
                  <option value="minimal">Minimal</option>
                  <option value="balanced">Balanced</option>
                  <option value="evidence">Evidence-heavy</option>
                  <option value="synthesis">Synthesis-heavy</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Approval mode</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={approvalMode}
                  onChange={(e) => setApprovalMode(e.target.value as any)}
                >
                  <option value="high_impact">High-impact only</option>
                  <option value="none">Disabled</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Execution mode</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={executionMode}
                  onChange={(e) => setExecutionMode(e.target.value as any)}
                >
                  <option value="plan_and_execute">Plan and execute</option>
                  <option value="adaptive">Adaptive</option>
                </select>
              </div>
            </div>
            <div className="rounded-lg border border-gray-200 p-3">
              <div className="flex items-center justify-between gap-2 mb-2">
                <label className="text-sm font-medium text-gray-700">Memory extraction policy</label>
                <label className="inline-flex items-center gap-2 text-xs text-gray-700">
                  <input
                    type="checkbox"
                    className="rounded border-gray-300"
                    checked={extractMemoryOnFailure}
                    onChange={(e) => setExtractMemoryOnFailure(e.target.checked)}
                  />
                  Extract on failure
                </label>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    Failed types (optional) {memoryFailedTypesPreview.length}/12
                  </label>
                  <textarea
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs font-mono"
                    rows={3}
                    value={memoryFailedTypesText}
                    onChange={(e) => setMemoryFailedTypesText(e.target.value)}
                    placeholder={'pattern\nlesson\ninsight'}
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    Completed types (optional) {memoryCompletedTypesPreview.length}/12
                  </label>
                  <textarea
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs font-mono"
                    rows={3}
                    value={memoryCompletedTypesText}
                    onChange={(e) => setMemoryCompletedTypesText(e.target.value)}
                    placeholder={'finding\nsummary'}
                  />
                </div>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Roles (one per line) {rolePreview.length}/12
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={5}
                value={rolesText}
                onChange={(e) => setRolesText(e.target.value)}
                placeholder={'researcher_documents\nresearcher_arxiv\nanalyst\nsynthesizer'}
              />
              <p className="mt-1 text-xs text-gray-500">
                Supported examples: researcher_documents, researcher_arxiv, analyst, synthesizer, monitor.
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Config overrides (optional JSON)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={3}
                value={configOverridesText}
                onChange={(e) => setConfigOverridesText(e.target.value)}
                placeholder='{"approval_checkpoints": {"iterations": [3]}}'
              />
            </div>
            <div className="flex justify-end gap-3 pt-4 border-t">
              <Button type="button" variant="secondary" onClick={() => onClose()}>
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={quickStartRoleWorkflowMutation.isLoading || rolePreview.length === 0}
              >
                {quickStartRoleWorkflowMutation.isLoading ? 'Starting...' : 'Start Workflow'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default QuickStartRoleWorkflowModal;
