/**
 * Quick start: a Claude-backend coding run.
 *
 * Module scope, like its siblings — declared in the page's render body it was
 * remounted on every page render, resetting the form below.
 */

import React, { useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';

import Button from '../common/Button';

/** Where this form remembers its last answers, between openings. */
const CLAUDE_QS_PREFS_KEY = 'autonomous_agents_claude_qs_prefs_v1';

interface QuickStartClaudeBackendModalProps {
  onClose: () => void;
  quickStartClaudeBackendMutation: any;
  codeSources: any[];
  templateRecommendGoal: string;
}

export const QuickStartClaudeBackendModal: React.FC<QuickStartClaudeBackendModalProps> = ({
  onClose,
  quickStartClaudeBackendMutation,
  codeSources,
  templateRecommendGoal,
}) => {

  const [name, setName] = useState(`Claude Backend Loop - ${new Date().toLocaleDateString()}`);
  const [goal, setGoal] = useState(templateRecommendGoal.trim() || 'Fix backend API tests and stabilize integrations');
  const [searchQuery, setSearchQuery] = useState('backend');
  const [selectedSourceId, setSelectedSourceId] = useState<string>('');
  const [filePathsText, setFilePathsText] = useState<string>('');
  const [commandsText, setCommandsText] = useState<string>('');
  const MAX_QS_COMMANDS = 6;
  const MAX_QS_FILE_PATHS = 32;

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(CLAUDE_QS_PREFS_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw || '{}') as Record<string, any>;
      const src = String(parsed?.source_id || '').trim();
      const q = String(parsed?.search_query || '').trim();
      const files = String(parsed?.file_paths_text || '');
      const cmds = String(parsed?.commands_text || '');
      if (src) setSelectedSourceId(src);
      if (q) setSearchQuery(q);
      if (files) setFilePathsText(files);
      if (cmds) setCommandsText(cmds);
    } catch {
      // Ignore malformed local preferences.
    }
  }, []);

  useEffect(() => {
    try {
      const payload = {
        source_id: selectedSourceId || '',
        search_query: searchQuery || '',
        file_paths_text: filePathsText || '',
        commands_text: commandsText || '',
      };
      window.localStorage.setItem(CLAUDE_QS_PREFS_KEY, JSON.stringify(payload));
    } catch {
      // Ignore localStorage write failures.
    }
  }, [selectedSourceId, searchQuery, filePathsText, commandsText]);

  useEffect(() => {
    if (!selectedSourceId) return;
    // An empty list means the sources have not loaded yet, not that the
    // remembered source is gone: validating against it clears a good id.
    if (codeSources.length === 0) return;
    const exists = codeSources.some((s: any) => String(s?.id || '') === String(selectedSourceId));
    if (!exists) {
      setSelectedSourceId('');
    }
  }, [codeSources, selectedSourceId]);

  const resetSavedDefaults = () => {
    try {
      window.localStorage.removeItem(CLAUDE_QS_PREFS_KEY);
    } catch {
      // Ignore localStorage failures.
    }
    setSelectedSourceId('');
    setSearchQuery('backend');
    setFilePathsText('');
    setCommandsText('');
    toast.success('Saved quick-start defaults reset');
  };

  const parseUniqueLines = (raw: string, maxItems: number): string[] => {
    const out: string[] = [];
    const seen = new Set<string>();
    const rows = String(raw || '')
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
    for (const row of rows) {
      if (seen.has(row)) continue;
      seen.add(row);
      out.push(row);
      if (out.length >= maxItems) break;
    }
    return out;
  };

  const parseSafeFilePaths = (raw: string, maxItems: number): { items: string[]; droppedUnsafe: number } => {
    const out: string[] = [];
    const seen = new Set<string>();
    const rows = String(raw || '')
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
    let droppedUnsafe = 0;
    for (const row of rows) {
      let path = String(row || '').replace(/\\/g, '/').trim();
      while (path.startsWith('./')) path = path.slice(2);
      if (!path || path.startsWith('/') || path.includes(':')) {
        droppedUnsafe += 1;
        continue;
      }
      const parts = path.split('/').filter((seg) => seg && seg !== '.');
      if (parts.some((seg) => seg === '..')) {
        droppedUnsafe += 1;
        continue;
      }
      const normalized = parts.join('/');
      if (!normalized || seen.has(normalized)) continue;
      seen.add(normalized);
      out.push(normalized);
      if (out.length >= maxItems) break;
    }
    return { items: out, droppedUnsafe };
  };

  const commandsPreview = useMemo(
    () => parseUniqueLines(commandsText, MAX_QS_COMMANDS),
    [commandsText]
  );
  const unsafeCommandsPreview = useMemo(() => {
    const blockedPatterns = [
      /\brm\s+-rf\b/i,
      /\bsudo\b/i,
      /\bmkfs\b/i,
      /\bdd\s+if=/i,
      /\bshutdown\b/i,
      /\breboot\b/i,
      /\bhalt\b/i,
      /\bpoweroff\b/i,
      /\bchown\b/i,
      /\bchmod\s+777\b/i,
    ];
    return commandsPreview.filter((cmd) => blockedPatterns.some((rx) => rx.test(String(cmd || '')))).slice(0, 6);
  }, [commandsPreview]);
  const filePathsParsed = useMemo(
    () => parseSafeFilePaths(filePathsText, MAX_QS_FILE_PATHS),
    [filePathsText]
  );
  const filePathsPreview = filePathsParsed.items;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim()) {
      toast.error('Goal is required');
      return;
    }
    if (!selectedSourceId) {
      toast.error('Target code source is required');
      return;
    }
    const commands = commandsPreview;
    const filePaths = filePathsPreview;

    const rawCommandRows = commandsText.split('\n').map((s) => s.trim()).filter(Boolean).length;
    const rawFileRows = filePathsText.split('\n').map((s) => s.trim()).filter(Boolean).length;
    if (rawCommandRows > commands.length || rawFileRows > filePaths.length) {
      toast('Quick Start normalized duplicate/extra rows to backend limits');
    }
    if (filePathsParsed.droppedUnsafe > 0) {
      toast(`Dropped ${filePathsParsed.droppedUnsafe} unsafe file path row(s)`);
    }
    if (unsafeCommandsPreview.length > 0) {
      toast.error(`Blocked unsafe command(s): ${unsafeCommandsPreview.join(' | ')}`);
      return;
    }

    quickStartClaudeBackendMutation.mutate({
      name: name.trim() || undefined,
      goal: goal.trim(),
      source_id: selectedSourceId,
      search_query: searchQuery.trim() || undefined,
      file_paths: filePaths.length ? filePaths : undefined,
      commands: commands.length ? commands : undefined,
      start_immediately: true,
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-1">Quick Start Claude Backend Loop</h2>
          <p className="text-sm text-gray-500 mb-4">
            Launch a code patch + verify + refine loop with minimal inputs.
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
                placeholder="Fix backend API tests and stabilize integrations"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Target code source</label>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={selectedSourceId}
                onChange={(e) => setSelectedSourceId(e.target.value)}
              >
                <option value="">Select a git source…</option>
                {codeSources.map((s: any) => (
                  <option key={String(s.id)} value={String(s.id)}>
                    {String(s.name || s.id)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Search Query (optional)</label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="backend"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Commands (optional, one per line) {commandsPreview.length}/{MAX_QS_COMMANDS}
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={4}
                value={commandsText}
                onChange={(e) => setCommandsText(e.target.value)}
                placeholder={'python -m pytest -q\nnpm test'}
              />
              <p className="mt-1 text-xs text-gray-500">
                Leave empty to auto-infer verification commands from project profile.
              </p>
              {unsafeCommandsPreview.length > 0 && (
                <p className="mt-1 text-xs text-rose-700">
                  Unsafe command(s) blocked: {unsafeCommandsPreview.join(' | ')}
                </p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Focused file paths (optional, one per line) {filePathsPreview.length}/{MAX_QS_FILE_PATHS}
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={3}
                value={filePathsText}
                onChange={(e) => setFilePathsText(e.target.value)}
                placeholder={'backend/app/api/endpoints/agent_jobs.py\nbackend/tests/test_agent_jobs_quick_start.py'}
              />
              <p className="mt-1 text-xs text-gray-500">
                Limit patch context to specific files when you already know the target area.
              </p>
              {filePathsParsed.droppedUnsafe > 0 && (
                <p className="mt-1 text-xs text-amber-700">
                  {filePathsParsed.droppedUnsafe} unsafe path row(s) ignored (absolute/parent-traversal/drive paths).
                </p>
              )}
            </div>
            <div className="flex justify-between items-center pt-1">
              <button
                type="button"
                className="text-xs text-gray-600 hover:text-gray-800 underline"
                onClick={resetSavedDefaults}
              >
                Reset saved defaults
              </button>
            </div>
            <div className="flex justify-end gap-3 pt-4 border-t">
              <Button type="button" variant="secondary" onClick={() => onClose()}>
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={quickStartClaudeBackendMutation.isLoading || codeSources.length === 0 || unsafeCommandsPreview.length > 0}
                title={
                  codeSources.length === 0
                    ? 'Add a GitHub/GitLab source first'
                    : unsafeCommandsPreview.length > 0
                      ? 'Remove unsafe commands before starting'
                      : undefined
                }
              >
                {quickStartClaudeBackendMutation.isLoading ? 'Starting...' : 'Start Loop'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default QuickStartClaudeBackendModal;
